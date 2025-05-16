# gum.py

from __future__ import annotations

import asyncio
import json
import logging
import os
from uuid import uuid4
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from importlib.resources import files as get_package_file
from typing import Callable, List
from .models import observation_proposition

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from .db_utils import (
    get_related_observations,
    search_propositions_bm25,
)
from .models import Observation, Proposition, init_db
from .observers import Observer
from .schemas import (
    PropositionItem,
    PropositionSchema,
    RelationSchema,
    Update,
    get_schema,
    AuditSchema
)

client = AsyncOpenAI()

class gum:
    def __init__(
        self,
        user_name: str,
        *observers: Observer,
        propose_prompt: str | None = None,
        similar_prompt: str | None = None,
        revise_prompt: str | None = None,
        audit_prompt: str | None = None,
        data_directory: str = "~/.cache/gum",
        db_name: str = "gum.db",
        max_concurrent_updates: int = 4,
        verbosity: int = logging.INFO,
        audit_enabled: bool = False,
    ):
        # basic paths
        data_directory = os.path.expanduser(data_directory)
        os.makedirs(data_directory, exist_ok=True)

        # runtime
        self.user_name = user_name
        self.observers: list[Observer] = list(observers)
        self.model = "gpt-4.1"
        self.audit_enabled = audit_enabled

        # logging
        self.logger = logging.getLogger("gum")
        self.logger.setLevel(verbosity)
        if not self.logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(h)

        # prompts
        self.propose_prompt = propose_prompt or self._load_prompt("propose.txt")
        self.similar_prompt = similar_prompt or self._load_prompt("similar.txt")
        self.revise_prompt = revise_prompt or self._load_prompt("revise.txt")
        self.audit_prompt = audit_prompt or self._load_prompt("audit.txt")

        self.engine = None
        self.Session = None
        self._db_name        = db_name
        self._data_directory = data_directory

        self._update_sem = asyncio.Semaphore(max_concurrent_updates)
        self._tasks: set[asyncio.Task] = set()
        self._loop_task: asyncio.Task | None = None
        self.update_handlers: list[Callable[[Observer, Update], None]] = []

    def start_update_loop(self):
        if self._loop_task is None:
            self._loop_task = asyncio.create_task(self._update_loop())

    async def stop_update_loop(self):
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None

    async def connect_db(self):
        if self.engine is None:
            self.engine, self.Session = await init_db(
                self._db_name, self._data_directory
            )

    async def __aenter__(self):
        await self.connect_db()
        self.start_update_loop()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.stop_update_loop()

        # wait for any in-flight handlers
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # stop observers
        for obs in self.observers:
            await obs.stop()

    async def _update_loop(self):
        """
        Efficiently wait for *any* observer to produce an Update and
        dispatch it through the semaphore-guarded handler.
        """
        while True:

            gets = {
                asyncio.create_task(obs.update_queue.get()): obs
                for obs in self.observers
            }

            done, _ = await asyncio.wait(
                gets.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for fut in done:
                upd: Update = fut.result()
                obs = gets[fut]

                t = asyncio.create_task(self._run_with_gate(obs, upd))
                self._tasks.add(t)

    async def _run_with_gate(self, observer: Observer, update: Update):
        """Wrapper that enforces max_concurrent_updates."""
        async with self._update_sem:
            try:
                await self._default_handler(observer, update)
            finally:
                self._tasks.discard(asyncio.current_task())

    async def _construct_propositions(self, update: Update) -> list[PropositionItem]:
        prompt = (
            self.propose_prompt.replace("{user_name}", self.user_name)
            .replace("{inputs}", update.content)
        )

        schema = PropositionSchema.model_json_schema()
        rsp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=get_schema(schema),
        )

        return json.loads(rsp.choices[0].message.content)["propositions"]

    async def _build_relation_prompt(self, all_props) -> str:
        blocks = [
            f"[id={p['id']}] {p['proposition']}\n    Reasoning: {p['reasoning']}"
            for p in all_props
        ]
        body = "\n\n".join(blocks)
        return self.similar_prompt.replace("{body}", body)

    async def _filter_propositions(
        self, rel_props: list[Proposition]
    ) -> tuple[list[Proposition], list[Proposition], list[Proposition]]:
        if not rel_props:
            return [], [], []

        payload = [
            {"id": p.id, "proposition": p.text, "reasoning": p.reasoning or ""}
            for p in rel_props
        ]
        prompt_text = await self._build_relation_prompt(payload)

        rsp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt_text}],
            response_format=get_schema(RelationSchema.model_json_schema()),
        )

        data = RelationSchema.model_validate_json(rsp.choices[0].message.content)

        id_to_prop = {p.id: p for p in rel_props}
        ident, sim, unrel = set(), set(), set()

        for r in data.relations:
            if r.label == "IDENTICAL":
                ident.add(r.source)
                ident.update(r.target or [])
            elif r.label == "SIMILAR":
                sim.add(r.source)
                sim.update(r.target or [])
            else:
                unrel.add(r.source)

        # only keep IDs we actually know about
        valid_ids = set(id_to_prop.keys())
        ident &= valid_ids
        sim &= valid_ids
        unrel &= valid_ids

        return (
            [id_to_prop[i] for i in ident],
            [id_to_prop[i] for i in sim - ident],
            [id_to_prop[i] for i in unrel - ident - sim],
        )


    async def _build_revision_body(
        self, similar: List[Proposition], related_obs: List[Observation]
    ) -> str:
        blocks = [
            f"Proposition {idx}: {p.text}\nReasoning: {p.reasoning}"
            for idx, p in enumerate(similar, 1)
        ]
        if related_obs:
            blocks.append("\nSupporting observations:")
            blocks.extend(f"- {o.content}" for o in related_obs[:10])
        return "\n".join(blocks)

    async def _revise_propositions(
        self,
        related_obs: list[Observation],
        similar_cluster: list[Proposition],
    ) -> list[dict]:
        body = await self._build_revision_body(similar_cluster, related_obs)
        prompt = self.revise_prompt.replace("{body}", body)
        rsp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=get_schema(PropositionSchema.model_json_schema()), 
        )
        return json.loads(rsp.choices[0].message.content)["propositions"]

    async def _generate_and_search(
        self, session: AsyncSession, update: Update, obs: Observation
    ) -> list[Proposition]:

        drafts_raw = await self._construct_propositions(update)
        drafts: list[Proposition] = []
        pool: dict[int, Proposition] = {}

        for itm in drafts_raw:
            draft = Proposition(
                text=itm["proposition"],
                reasoning=itm["reasoning"],
                confidence=itm.get("confidence"),
                decay=itm.get("decay"),
                revision_group=str(uuid4()),
                version=1,
            )
            drafts.append(draft)

            # search existing persisted props
            with session.no_autoflush:
                hits = await search_propositions_bm25(
                    session, f"{draft.text}\n{draft.reasoning}", mode="OR"
                )
            for prop, _score in hits:
                pool[prop.id] = prop

        session.add_all(drafts)
        await session.flush()

        for draft in drafts:
            pool[draft.id] = draft

        return list(pool.values())

    async def _handle_identical(
        self, session, identical: list[Proposition], obs: Observation
    ) -> None:
        for p in identical:
            await self._attach_obs_if_missing(p, obs, session)

    async def _handle_similar(
        self,
        session: AsyncSession,
        similar: list[Proposition],
        obs: Observation,
    ) -> None:

        if not similar:
            return

        rel_obs = {
            o
            for p in similar
            for o in await get_related_observations(session, p.id)
        }
        rel_obs.add(obs)

        revised_items = await self._revise_propositions(list(rel_obs), similar)
        newest_version = max(p.version for p in similar)
        parent_groups = {p.revision_group for p in similar}
        if len(parent_groups) == 1:
            revision_group = parent_groups.pop()
        else:
            revision_group = uuid4().hex

        new_children: list[Proposition] = []
        for item in revised_items:
            child = Proposition(
                text=item["proposition"],
                reasoning=item["reasoning"],
                confidence=item.get("confidence"),
                decay=item.get("decay"),
                version=newest_version + 1,
                revision_group=revision_group,
                observations=rel_obs,
                parents=set(similar),
            )
            session.add(child)
            new_children.append(child)

        await session.flush()

    async def _handle_different(
        self, session, different: list[Proposition], obs: Observation
    ) -> None:
        for p in different:
            await self._attach_obs_if_missing(p, obs, session)

    async def _handle_audit(self, obs: Observation) -> bool:
        if not self.audit_enabled:
            return False

        hits = await self.query(obs.content, limit=10, mode="OR")

        if not hits:
            past_interaction = "*None*"
        else:
            ctx_chunks: list[str] = []
            async with self._session() as session:
                for prop, score in hits:
                    chunk = [f"• {prop.text}"]
                    if prop.reasoning:
                        chunk.append(f"  Reasoning: {prop.reasoning}")
                    if prop.confidence is not None:
                        chunk.append(f"  Confidence: {prop.confidence}")
                    chunk.append(f"  Relevance Score: {score:.2f}")

                    obs_list = await get_related_observations(session, prop.id)
                    if obs_list:
                        chunk.append("  Supporting Observations:")
                        for rel_obs in obs_list:
                            preview = rel_obs.content.replace("\n", " ")[:120]
                            chunk.append(f"    - [{rel_obs.observer_name}] {preview}")

                    ctx_chunks.append("\n".join(chunk))

            past_interaction = "\n\n".join(ctx_chunks)

        prompt = (
            self.audit_prompt
            .replace("{past_interaction}", past_interaction)
            .replace("{user_input}", obs.content)
            .replace("{user_name}", self.user_name)
        )

        rsp = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=get_schema(AuditSchema.model_json_schema()),
            temperature=0.0,
        )
        decision = json.loads(rsp.choices[0].message.content)

        if not decision["transmit_data"]:
            self.logger.warning(
                "Audit blocked transmission (data_type=%s, subject=%s)",
                decision["data_type"],
                decision["subject"],
            )
            return True

        return False

    async def _default_handler(self, observer: Observer, update: Update) -> None:
        self.logger.info(f"Processing update from {observer.name}")

        async with self._session() as session:
            observation = Observation(
                observer_name=observer.name,
                content=update.content,
                content_type=update.content_type,
            )

            if await self._handle_audit(observation):
                return

            session.add(observation)
            await session.flush()

            pool = await self._generate_and_search(session, update, observation)
            identical, similar, different = await self._filter_propositions(pool)

            self.logger.info("Applying proposition updates...")
            await self._handle_identical(session, identical, observation)
            await self._handle_similar(session, similar, observation)
            await self._handle_different(session, different, observation)
            self.logger.info("Completed processing update")

    @asynccontextmanager
    async def _session(self):
        async with self.Session() as s:
            async with s.begin():
                yield s

    @staticmethod
    async def _attach_obs_if_missing(prop: Proposition, obs: Observation, session):
        await session.execute(
            insert(observation_proposition)
            .prefix_with("OR IGNORE")
            .values(observation_id=obs.id, proposition_id=prop.id)
        )
        prop.updated_at = datetime.now(timezone.utc)

    @staticmethod
    def _load_prompt(fname: str) -> str:
        return get_package_file("gum.prompts.gum").joinpath(fname).read_text()

    def add_observer(self, observer: Observer):
        self.observers.append(observer)

    def remove_observer(self, observer: Observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def register_update_handler(self, fn: Callable[[Observer, Update], None]):
        self.update_handlers.append(fn)

    async def query(
        self,
        user_query: str,
        *,
        limit: int = 3,
        mode: str = "OR",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[tuple[Proposition, float]]:
        """Query propositions using BM25 search with optional time range filtering.

        Args:
            user_query: Search query string
            limit: Maximum number of results to return
            mode: Search mode ("AND", "OR", or "PHRASE")
            start_time: Start of time range (UTC, inclusive)
            end_time: End of time range (UTC, inclusive, defaults to now)
        """
        async with self._session() as session:
            return await search_propositions_bm25(
                session,
                user_query,
                limit=limit,
                mode=mode,
                start_time=start_time,
                end_time=end_time,
            )
