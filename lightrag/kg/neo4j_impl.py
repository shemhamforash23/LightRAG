import asyncio
import configparser
import logging
import os
import random
import re
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple, final

import numpy as np
import pipmaster as pm
from opentelemetry import context, trace
from opentelemetry.trace import Status, StatusCode
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.lightrag import tracer

from ..base import BaseGraphStorage
from ..types import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode
from ..utils import logger

if not pm.is_installed("neo4j"):
    pm.install("neo4j")

from neo4j import (  # type: ignore
    AsyncDriver,
    AsyncGraphDatabase,
    AsyncManagedTransaction,
    GraphDatabase,
)
from neo4j import (
    exceptions as neo4jExceptions,
)

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")

# Get maximum number of graph nodes from environment variable, default is 1000
MAX_GRAPH_NODES = int(os.getenv("MAX_GRAPH_NODES", 1000))

# Default values for timeouts and thresholds
DEFAULT_NEO4J_OPERATION_TIMEOUT = float(
    os.environ.get("NEO4J_OPERATION_TIMEOUT", 30)
)  # seconds
DEFAULT_LONG_TX_THRESHOLD = float(
    os.environ.get("NEO4J_LONG_TX_THRESHOLD", 10)
)  # seconds

# Default values for queue and worker pool
DEFAULT_NEO4J_WORKER_COUNT = int(
    os.environ.get("NEO4J_WORKER_COUNT", 10)
)  # number of workers
DEFAULT_NEO4J_QUEUE_SIZE = int(
    os.environ.get("NEO4J_QUEUE_SIZE", 1000)
)  # max queue size
DEFAULT_NEO4J_DEFAULT_PRIORITY = int(
    os.environ.get("NEO4J_DEFAULT_PRIORITY", 100)
)  # default priority

# Set neo4j logger level to ERROR to suppress warning logs
logging.getLogger("neo4j").setLevel(logging.ERROR)


class Neo4jTask:
    """
    Class representing a task in the Neo4j request queue.
    """

    def __init__(
        self,
        func: Callable,
        instance: Any,
        args: Tuple,
        kwargs: Dict,
        priority: int = DEFAULT_NEO4J_DEFAULT_PRIORITY,
        parent_context: Optional[context.Context] = None,
    ):
        self.id = str(uuid.uuid4())
        self.func = func
        self.instance = instance
        self.args = args
        self.kwargs = kwargs
        self.priority = priority
        self.parent_context = parent_context
        self.future: asyncio.Future = asyncio.Future()
        self.start_time = None
        self.end_time = None

    def __lt__(self, other):
        # For sorting in PriorityQueue
        return self.priority < other.priority


async def neo4j_worker(
    worker_id: int, task_queue: asyncio.PriorityQueue, semaphore: asyncio.Semaphore
):
    """
    Worker function for processing tasks from the Neo4j request queue.

    Args:
        worker_id: Worker identifier
        task_queue: Task queue
        semaphore: Semaphore for limiting the number of concurrent requests
    """
    logger.info(f"Neo4j worker {worker_id} started")

    while True:
        try:
            # Get task from queue
            priority_value, task = await task_queue.get()

            # Log task received
            logger.debug(
                f"Worker {worker_id}: Received task {task.id} with priority {priority_value}, "
                f"function: {task.func.__name__}, args: {task.args}"
            )

            # Set start time
            task.start_time = time.time()

            # Start OpenTelemetry Span
            operation_name = f"neo4j.{task.func.__name__}"

            # Start OpenTelemetry span using the parent context from the task
            with tracer.start_as_current_span(
                operation_name, kind=trace.SpanKind.CLIENT, context=task.parent_context
            ) as span:
                span.set_attribute("db.system", "neo4j")
                span.set_attribute("db.operation", task.func.__name__)
                # Consider security before adding args/kwargs
                span.set_attribute(
                    "db.statement", f"args: {task.args}, kwargs: {task.kwargs}"
                )
                span.set_attribute("neo4j.task.id", task.id)
                span.set_attribute("neo4j.task.priority", priority_value)
                span.set_attribute("neo4j.worker.id", worker_id)

                try:
                    # Get timeout from object attribute or use default value
                    operation_timeout = getattr(
                        task.instance,
                        "NEO4J_OPERATION_TIMEOUT",
                        DEFAULT_NEO4J_OPERATION_TIMEOUT,
                    )
                    long_tx_threshold = getattr(
                        task.instance,
                        "LONG_TRANSACTION_THRESHOLD",
                        DEFAULT_LONG_TX_THRESHOLD,
                    )

                    # Get function name for logging
                    operation_name = (
                        f"{task.func.__name__}({', '.join(map(str, task.args))})"
                    )

                    # Start monitoring for long transaction
                    monitor_task = asyncio.create_task(
                        monitor_long_transaction(long_tx_threshold, operation_name)
                    )

                    try:
                        span.add_event("Waiting for semaphore")
                        async with semaphore:
                            span.add_event("Acquired semaphore")

                            # Add random delay if system is under high load
                            if random.random() < 0.3:  # 30% chance to add delay
                                delay = random.uniform(0.05, 0.2)
                                span.add_event(
                                    "Adding artificial delay", {"duration_s": delay}
                                )
                                await asyncio.sleep(delay)

                            # Execute operation with timeout
                            start_time = time.time()
                            span.add_event("Executing Neo4j function")
                            result = await asyncio.wait_for(
                                task.func(task.instance, *task.args, **task.kwargs),
                                timeout=operation_timeout,
                            )
                            execution_time = time.time() - start_time
                            span.set_attribute("neo4j.execution_time_s", execution_time)
                            span.add_event("Neo4j function executed")

                            # Set result in future
                            task.future.set_result(result)
                            span.set_status(Status(StatusCode.OK))

                    except asyncio.TimeoutError as e:
                        logger.error(
                            f"Worker {worker_id}: Operation {operation_name} timed out after {operation_timeout} seconds"
                        )
                        span.record_exception(e)
                        span.set_status(
                            Status(
                                StatusCode.ERROR, f"Timeout after {operation_timeout}s"
                            )
                        )
                        task.future.set_exception(
                            RuntimeError(
                                f"Neo4j operation {operation_name} timed out after {operation_timeout} seconds"
                            )
                        )
                    except Exception as e:
                        logger.error(
                            f"Worker {worker_id}: Error in Neo4j operation {operation_name}: {str(e)}"
                        )
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        task.future.set_exception(e)
                    finally:
                        # Cancel monitoring task
                        if not monitor_task.done():
                            monitor_task.cancel()
                            span.add_event("Cancelled long transaction monitor")

                except Exception as e:
                    logger.error(
                        f"Worker {worker_id}: Error processing task {task.id}: {str(e)}"
                    )
                    # Ensure span records the error if setup fails before the inner try block
                    if trace.get_current_span().is_recording():
                        span.record_exception(e)
                        span.set_status(
                            Status(StatusCode.ERROR, f"Task processing error: {str(e)}")
                        )
                    if not task.future.done():
                        task.future.set_exception(e)

                finally:
                    # Set end time
                    task.end_time = time.time()
                    task_duration = task.end_time - task.start_time
                    span.set_attribute("neo4j.task.total_duration_s", task_duration)

                    # Log task completion
                    logger.debug(
                        f"Worker {worker_id}: Task {task.id} finished in {task_duration:.3f}s, "
                        f"queue size: {task_queue.qsize()}/{task_queue.maxsize}"
                    )

                    # Mark task as done
                    task_queue.task_done()

        except asyncio.CancelledError:
            logger.info(f"Neo4j worker {worker_id} cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in Neo4j worker {worker_id}: {str(e)}")
            # Continue worker operation even with unexpected errors


# Decorator for using semaphore and request queue
def with_semaphore(func):
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        # Check if workers are started
        if not self._workers_started:
            raise RuntimeError(
                "Neo4j workers are not started. Make sure initialize() was called."
            )

        # Get priority from kwargs or use default value
        priority = kwargs.pop("priority", DEFAULT_NEO4J_DEFAULT_PRIORITY)

        # Capture the current OpenTelemetry context
        current_context = context.get_current()

        # Create task and pass the captured context
        task = Neo4jTask(
            func, self, args, kwargs, priority, parent_context=current_context
        )

        # Add task to queue
        await self._task_queue.put((priority, task))

        # Wait for task completion
        return await task.future

    return wrapper


async def monitor_long_transaction(threshold: float, operation_name: str):
    """
    Function for monitoring long-running transactions.
    If the operation doesn't complete within threshold seconds, a warning is logged.

    Args:
        threshold: Time in seconds after which the transaction is considered long-running
        operation_name: Name of the operation being monitored
    """
    await asyncio.sleep(threshold)
    logger.warning(
        f"Long-running transaction detected: {operation_name} is running for more than {threshold} seconds"
    )


@final
@dataclass
class Neo4JStorage(BaseGraphStorage):
    def __init__(self, namespace, global_config, embedding_func):
        super().__init__(
            namespace=namespace,
            global_config=global_config,
            embedding_func=embedding_func,
        )
        self._driver_lock = asyncio.Lock()

        # Get maximum number of concurrent requests from environment variable
        self.NEO4J_MAX_CONCURRENT_REQUESTS = int(
            os.environ.get(
                "NEO4J_MAX_CONCURRENT_REQUESTS",
                config.get("neo4j", "max_concurrent_requests", fallback=10),
            )
        )
        # Initialize semaphore with specified size
        self._semaphore = asyncio.Semaphore(self.NEO4J_MAX_CONCURRENT_REQUESTS)

        # Set timeouts and thresholds for long transactions
        self.NEO4J_OPERATION_TIMEOUT = float(
            os.environ.get(
                "NEO4J_OPERATION_TIMEOUT",
                config.get(
                    "neo4j",
                    "operation_timeout",
                    fallback=DEFAULT_NEO4J_OPERATION_TIMEOUT,
                ),
            )
        )
        self.LONG_TRANSACTION_THRESHOLD = float(
            os.environ.get(
                "NEO4J_LONG_TX_THRESHOLD",
                config.get(
                    "neo4j", "long_tx_threshold", fallback=DEFAULT_LONG_TX_THRESHOLD
                ),
            )
        )

        # Initialize task queue and worker pool
        self.NEO4J_WORKER_COUNT = int(
            os.environ.get(
                "NEO4J_WORKER_COUNT",
                config.get(
                    "neo4j", "worker_count", fallback=DEFAULT_NEO4J_WORKER_COUNT
                ),
            )
        )
        self.NEO4J_QUEUE_SIZE = int(
            os.environ.get(
                "NEO4J_QUEUE_SIZE",
                config.get("neo4j", "queue_size", fallback=DEFAULT_NEO4J_QUEUE_SIZE),
            )
        )

        # Create task queue
        self._task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.NEO4J_QUEUE_SIZE
        )

        # Initialize workers list but don't start them yet
        self._workers = []
        self._workers_started = False

        logger.info(
            f"Initialized Neo4j storage with queue size {self.NEO4J_QUEUE_SIZE}"
        )

        self.URI = os.environ.get("NEO4J_URI", config.get("neo4j", "uri", fallback=""))
        self.USERNAME = os.environ.get(
            "NEO4J_USERNAME", config.get("neo4j", "username", fallback="")
        )
        self.PASSWORD = os.environ.get(
            "NEO4J_PASSWORD", config.get("neo4j", "password", fallback="")
        )
        self.MAX_CONNECTION_POOL_SIZE = int(
            os.environ.get(
                "NEO4J_MAX_CONNECTION_POOL_SIZE",
                config.get("neo4j", "connection_pool_size", fallback=100),
            )
        )
        self.CONNECTION_TIMEOUT = float(
            os.environ.get(
                "NEO4J_CONNECTION_TIMEOUT",
                config.get("neo4j", "connection_timeout", fallback=30.0),
            ),
        )
        self.CONNECTION_ACQUISITION_TIMEOUT = float(
            os.environ.get(
                "NEO4J_CONNECTION_ACQUISITION_TIMEOUT",
                config.get("neo4j", "connection_acquisition_timeout", fallback=30.0),
            ),
        )
        self.MAX_TRANSACTION_RETRY_TIME = float(
            os.environ.get(
                "NEO4J_MAX_TRANSACTION_RETRY_TIME",
                config.get("neo4j", "max_transaction_retry_time", fallback=30.0),
            ),
        )
        self.DATABASE = os.environ.get(
            "NEO4J_DATABASE", re.sub(r"[^a-zA-Z0-9-]", "-", namespace)
        )

        self._driver: AsyncDriver = AsyncGraphDatabase.driver(
            self.URI,
            auth=(self.USERNAME, self.PASSWORD),
            max_connection_pool_size=self.MAX_CONNECTION_POOL_SIZE,
            connection_timeout=self.CONNECTION_TIMEOUT,
            connection_acquisition_timeout=self.CONNECTION_ACQUISITION_TIMEOUT,
            max_transaction_retry_time=self.MAX_TRANSACTION_RETRY_TIME,
        )

    def __post_init__(self):
        self._node_embed_algorithms = {
            "node2vec": self._node2vec_embed,
        }

    async def close(self):
        """Close the Neo4j driver and release all resources"""
        # Stop all worker tasks first
        await self.stop_workers()

        # Then close the driver
        if self._driver:
            await self._driver.close()

    async def __aexit__(self, exc_type, exc, tb):
        """Ensure driver is closed when context manager exits"""
        await self.close()

    async def index_done_callback(self) -> None:
        # Noe4J handles persistence automatically
        pass

    async def initialize(self) -> None:
        """Initialize the Neo4j storage."""
        # Try to connect to the database
        with GraphDatabase.driver(
            self.URI,
            auth=(self.USERNAME, self.PASSWORD),
            max_connection_pool_size=self.MAX_CONNECTION_POOL_SIZE,
            connection_timeout=self.CONNECTION_TIMEOUT,
            connection_acquisition_timeout=self.CONNECTION_ACQUISITION_TIMEOUT,
        ) as _sync_driver:
            for database in (self.DATABASE, None):
                self._DATABASE = database
                connected = False

                try:
                    with _sync_driver.session(database=database) as session:
                        try:
                            session.run("MATCH (n) RETURN n LIMIT 0")
                            logger.info(f"Connected to {database} at {self.URI}")
                            connected = True
                        except neo4jExceptions.ServiceUnavailable as e:
                            logger.error(
                                f"{database} at {self.URI} is not available".capitalize()
                            )
                            raise e
                except neo4jExceptions.AuthError as e:
                    logger.error(f"Authentication failed for {database} at {self.URI}")
                    raise e
                except neo4jExceptions.ClientError as e:
                    if e.code == "Neo.ClientError.Database.DatabaseNotFound":
                        logger.info(
                            f"{database} at {self.URI} not found. Try to create specified database.".capitalize()
                        )
                        try:
                            with _sync_driver.session() as session:
                                session.run(
                                    f"CREATE DATABASE `{database}` IF NOT EXISTS"
                                )
                                logger.info(
                                    f"{database} at {self.URI} created".capitalize()
                                )
                                connected = True
                        except (
                            neo4jExceptions.ClientError,
                            neo4jExceptions.DatabaseError,
                        ) as e:
                            if (
                                e.code
                                == "Neo.ClientError.Statement.UnsupportedAdministrationCommand"
                            ) or (
                                e.code == "Neo.DatabaseError.Statement.ExecutionFailed"
                            ):
                                if database is not None:
                                    logger.warning(
                                        "This Neo4j instance does not support creating databases. Try to use Neo4j Desktop/Enterprise version or DozerDB instead. Fallback to use the default database."
                                    )
                            if database is None:
                                logger.error(
                                    f"Failed to create {database} at {self.URI}"
                                )
                                raise e

                if connected:
                    break

        # Start worker tasks after successful database connection
        await self.start_workers()
        logger.info(f"Started {self.NEO4J_WORKER_COUNT} Neo4j worker tasks")

    async def finalize(self) -> None:
        """Finalize the Neo4j storage."""
        await self.close()

    async def drop(self) -> None:
        """Drop all data from the Neo4j storage.

        This method removes all nodes and relationships from the graph database.
        """
        try:
            # Get all node labels
            labels = await self.get_all_labels()

            if labels:
                # Remove all nodes (this will also remove all relationships)
                await self.remove_nodes(labels)
                logger.info(
                    f"Successfully cleared all data from Neo4j database {self._DATABASE}"
                )
            else:
                logger.info(f"No data to clear from Neo4j database {self._DATABASE}")

        except Exception as e:
            logger.error(f"Error while clearing Neo4j database: {e}")
            raise

    @with_semaphore
    async def has_node(self, node_id: str) -> bool:
        """
        Check if a node with the given label exists in the database

        Args:
            node_id: Label of the node to check

        Returns:
            bool: True if node exists, False otherwise

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = "MATCH (n:base {entity_id: $entity_id}) RETURN count(n) > 0 AS node_exists"
                result = await session.run(query, entity_id=node_id)
                single_result = await result.single()
                assert (
                    single_result is not None
                ), f"Failed to check node existence for {node_id}"
                await result.consume()  # Ensure result is fully consumed
                return single_result["node_exists"]
            except Exception as e:
                logger.error(f"Error checking node existence for {node_id}: {str(e)}")
                await result.consume()  # Ensure results are consumed even on error
                raise

    @with_semaphore
    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """
        Check if an edge exists between two nodes

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            bool: True if edge exists, False otherwise

        Raises:
            ValueError: If either node_id is invalid
            Exception: If there is an error executing the query
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                query = (
                    "MATCH (a:base {entity_id: $source_entity_id})-[r]-(b:base {entity_id: $target_entity_id}) "
                    "RETURN COUNT(r) > 0 AS edgeExists"
                )
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                single_result = await result.single()
                assert (
                    single_result is not None
                ), f"Failed to check edge existence between {source_node_id} and {target_node_id}"
                await result.consume()  # Ensure result is fully consumed
                return single_result["edgeExists"]
            except Exception as e:
                logger.error(
                    f"Error checking edge existence between {source_node_id} and {target_node_id}: {str(e)}"
                )
                await result.consume()  # Ensure results are consumed even on error
                raise

    @with_semaphore
    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier.

        Args:
            node_id: The node label to look up

        Returns:
            dict: Node properties if found
            None: If node not found or on error
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                try:
                    query = "MATCH (n:base {entity_id: $entity_id}) RETURN n"
                    result = await session.run(query, entity_id=node_id)
                    try:
                        records = await result.fetch(
                            2
                        )  # Get 2 records for duplication check

                        if len(records) > 1:
                            logger.warning(
                                f"Multiple nodes found with label '{node_id}'. Using first node."
                            )
                        if records:
                            node = records[0]["n"]
                            node_dict = dict(node)
                            # Remove base label from labels list if it exists
                            if "labels" in node_dict:
                                node_dict["labels"] = [
                                    label
                                    for label in node_dict["labels"]
                                    if label != "base"
                                ]
                            logger.debug(
                                f"Neo4j query node {query} return: {node_dict}"
                            )
                            return node_dict
                        return None
                    finally:
                        await result.consume()  # Ensure result is fully consumed
                except Exception as e:
                    logger.error(f"Error getting node for {node_id}: {str(e)}")
                    return None  # Return None instead of raising
        except Exception as e:
            logger.error(f"Session error in get_node for {node_id}: {str(e)}")
            return None  # Return None instead of raising

    @with_semaphore
    async def node_degree(self, node_id: str) -> int:
        """Get the degree (number of relationships) of a node with the given label.
        If multiple nodes have the same label, returns the degree of the first node.
        If no node is found, returns 0.

        Args:
            node_id: The label of the node

        Returns:
            int: The number of relationships the node has, or 0 if no node found

        Raises:
            ValueError: If node_id is invalid
            Exception: If there is an error executing the query
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                try:
                    query = """
                        MATCH (n:base {entity_id: $entity_id})
                        OPTIONAL MATCH (n)-[r]-()
                        RETURN COUNT(r) AS degree
                    """
                    result = await session.run(query, entity_id=node_id)
                    try:
                        record = await result.single()

                        if not record:
                            logger.warning(f"No node found with label '{node_id}'")
                            return 0

                        degree = record["degree"]
                        logger.debug(
                            f"Neo4j query node degree for {node_id} return: {degree}"
                        )
                        return degree
                    finally:
                        await result.consume()  # Ensure result is fully consumed
                except Exception as e:
                    logger.error(f"Error getting node degree for {node_id}: {str(e)}")
                    # Return 0 instead of raising to make the method more robust
                    return 0
        except Exception as e:
            logger.error(f"Session error getting node degree for {node_id}: {str(e)}")
            # Return 0 instead of raising to make the method more robust
            return 0

    @with_semaphore
    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get the total degree (sum of relationships) of two nodes.

        Args:
            src_id: Label of the source node
            tgt_id: Label of the target node

        Returns:
            int: Sum of the degrees of both nodes
        """
        try:
            # Use a single query to get both degrees to reduce the number of transactions
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                    MATCH (src:base {entity_id: $src_id})
                    OPTIONAL MATCH (src)-[r1]-()
                    WITH COUNT(r1) AS src_degree

                    MATCH (tgt:base {entity_id: $tgt_id})
                    OPTIONAL MATCH (tgt)-[r2]-()
                    WITH src_degree, COUNT(r2) AS tgt_degree

                    RETURN src_degree + tgt_degree AS total_degree
                """
                result = await session.run(query, src_id=src_id, tgt_id=tgt_id)
                try:
                    record = await result.single()

                    if not record:
                        logger.warning(
                            f"No nodes found for edge degree '{src_id}' to '{tgt_id}'"
                        )
                        return 0

                    total_degree = record["total_degree"]
                    logger.debug(
                        f"Neo4j query edge degree for {src_id}-{tgt_id} return: {total_degree}"
                    )
                    return total_degree
                finally:
                    await result.consume()  # Ensure result is fully consumed
        except Exception as e:
            logger.error(f"Error getting edge degree for {src_id}-{tgt_id}: {str(e)}")
            # Return 0 instead of raising to make the method more robust
            return 0

    @with_semaphore
    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get the properties of the edge between two nodes.

        Args:
            source_node_id: Label of the source node
            target_node_id: Label of the target node

        Returns:
            dict: Properties of the edge, or None if no edge found
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
                RETURN properties(r) as edge_properties
                """
                result = await session.run(
                    query,
                    source_entity_id=source_node_id,
                    target_entity_id=target_node_id,
                )
                try:
                    record = await result.single()
                    if record and record["edge_properties"]:
                        edge_properties = record["edge_properties"]
                        logger.debug(
                            f"get_edge:query:\n{query}\n:result:{edge_properties}"
                        )
                        # Convert all values to strings for consistent handling
                        return {
                            k: str(v) if v is not None else ""
                            for k, v in edge_properties.items()
                        }
                    else:
                        # Return default edge properties when no edge found
                        return {
                            "weight": str(0.0),
                            "source_id": "",
                            "description": "",
                            "keywords": "",
                        }
                except Exception as e:
                    logger.error(
                        f"Error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
                    )
                    # Return default edge properties on error
                    return {
                        "weight": str(0.0),
                        "source_id": "",
                        "description": "",
                        "keywords": "",
                    }
                finally:
                    await result.consume()  # Ensure result is fully consumed
        except Exception as e:
            logger.error(
                f"Session error in get_edge between {source_node_id} and {target_node_id}: {str(e)}"
            )
            # Return default edge properties on error
            return {
                "weight": str(0.0),
                "source_id": "",
                "description": "",
                "keywords": "",
            }

    @with_semaphore
    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """Retrieves all edges (relationships) for a particular node identified by its label.

        Args:
            source_node_id: Label of the node to get edges for

        Returns:
            list[tuple[str, str]]: List of (source_label, target_label) tuples representing edges
            None: If no edges found
        """
        try:
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                try:
                    query = """MATCH (n:base {entity_id: $entity_id})
                            OPTIONAL MATCH (n)-[r]-(connected:base)
                            WHERE connected.entity_id IS NOT NULL
                            RETURN n, r, connected"""
                    results = await session.run(query, entity_id=source_node_id)

                    edges = []
                    try:
                        async for record in results:
                            source_node = record["n"]
                            connected_node = record["connected"]

                            # Skip if either node is None
                            if not source_node or not connected_node:
                                continue

                            source_label = (
                                source_node.get("entity_id")
                                if source_node.get("entity_id")
                                else None
                            )
                            target_label = (
                                connected_node.get("entity_id")
                                if connected_node.get("entity_id")
                                else None
                            )

                            if source_label and target_label:
                                edges.append((source_label, target_label))

                        return edges
                    finally:
                        await results.consume()  # Ensure results are consumed
                except Exception as e:
                    logger.error(
                        f"Error getting edges for node {source_node_id}: {str(e)}"
                    )
                    return []  # Return empty list instead of raising
        except Exception as e:
            logger.error(
                f"Session error in get_node_edges for {source_node_id}: {str(e)}"
            )
            return []  # Return empty list instead of raising

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    @with_semaphore
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        properties = node_data
        entity_type = properties["entity_type"]
        entity_id = properties["entity_id"]
        if "entity_id" not in properties:
            raise ValueError("Neo4j: node properties must contain an 'entity_id' field")

        try:
            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    query = (
                        """
                    MERGE (n:base {entity_id: $properties.entity_id})
                    SET n += $properties
                    SET n:`%s`
                    """
                        % entity_type
                    )
                    result = await tx.run(query, properties=properties)
                    logger.debug(
                        f"Upserted node with entity_id '{entity_id}' and properties: {properties}"
                    )
                    await result.consume()  # Ensure result is fully consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(f"Error during upsert: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    @with_semaphore
    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        relationship_type: str,
        edge_data: dict = {},
    ) -> None:
        """
        Upsert an edge with a specific type and properties between two nodes identified by their entity_id.
        Ensures both source and target nodes exist before creating/updating the edge.
        Uses apoc.merge.relationship to handle dynamic relationship types.

        Args:
            source_node_id (str): entity_id of the source node.
            target_node_id (str): entity_id of the target node.
            relationship_type (str): The type of the relationship (e.g., 'CALLS', 'IMPLEMENTS'). Will be standardized to UPPER_SNAKE_CASE.
            edge_data (dict, optional): Dictionary of properties to set/update on the edge. Defaults to {}.

        Raises:
            Exception: Propagates exceptions from the database driver during the operation.
        """
        try:
            # Standardize relationship type
            std_relationship_type = relationship_type.strip().upper().replace(" ", "_")
            if not std_relationship_type:
                std_relationship_type = (
                    "RELATED_TO"  # Fallback if empty after processing
                )
                logger.warning(
                    f"Empty relationship type provided for edge ({source_node_id}, {target_node_id}). Using fallback '{std_relationship_type}'."
                )

            async with self._driver.session(database=self._DATABASE) as session:

                async def execute_upsert(tx: AsyncManagedTransaction):
                    # Using apoc.merge.relationship for dynamic relationship types
                    # Note: Assumes nodes have the label 'base' and are identified by 'entity_id'
                    # apoc.merge.relationship(startNode, relationshipType, identProps, props, endNode, onCreateProps)
                    query = """
                    MATCH (a:base {entity_id: $src_id}), (b:base {entity_id: $tgt_id})
                    CALL apoc.merge.relationship(
                        a,                    // start node
                        $rel_type,            // relationship type string
                        {},                   // properties to match on relationship (usually empty for upsert)
                        $props,               // properties to set/update on match
                        b,                    // end node
                        $props                // properties to set on create (same as on match here)
                    ) YIELD rel
                    RETURN rel
                    """
                    result = await tx.run(
                        query,
                        src_id=source_node_id,
                        tgt_id=target_node_id,
                        rel_type=std_relationship_type,  # Pass standardized type
                        props=edge_data,  # Pass edge data as properties
                    )
                    try:
                        records = await result.fetch(
                            1
                        )  # Expecting one record for the created/merged relationship
                        if records:
                            logger.debug(
                                f"Upserted edge '{std_relationship_type}' from '{source_node_id}' to '{target_node_id}' "
                                f"with properties: {edge_data}"
                            )
                        else:
                            # This case might indicate an issue, e.g., source or target node not found despite the MATCH
                            # Although MATCH should fail if nodes don't exist, apoc might behave differently or there's a tx issue.
                            logger.warning(
                                f"apoc.merge.relationship did not return a relationship for edge ({source_node_id})-[:{std_relationship_type}]->({target_node_id}). Nodes might not exist or another issue occurred."
                            )
                    finally:
                        await result.consume()  # Ensure result is consumed

                await session.execute_write(execute_upsert)
        except Exception as e:
            logger.error(
                f"Error during edge upsert ({source_node_id})-[:{relationship_type}]->({target_node_id}): {e}",
                exc_info=True,  # Include stack trace
            )
            raise

    @with_semaphore
    async def _node2vec_embed(self):
        print("Implemented but never called.")

    @with_semaphore
    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        min_degree: int = 0,
        inclusive: bool = False,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. min_degree does not affect nodes directly connected to the matching nodes
            2. Label matching nodes take precedence
            3. Followed by nodes directly connected to the matching nodes
            4. Finally, the degree of the nodes

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            min_degree: Minimum degree of nodes to include. Defaults to 0
            inclusive: Do an inclusive search if true
        Returns:
            KnowledgeGraph: Complete connected subgraph for specified node
        """
        result = KnowledgeGraph()
        seen_nodes = set()
        seen_edges = set()

        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            try:
                if node_label == "*":
                    main_query = """
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]-()
                    WITH n, COALESCE(count(r), 0) AS degree
                    WHERE degree >= $min_degree
                    ORDER BY degree DESC
                    LIMIT $max_nodes
                    WITH collect({node: n}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                            collect(DISTINCT r) AS relationships
                    """
                    result_set = await session.run(
                        main_query,
                        {"max_nodes": MAX_GRAPH_NODES, "min_degree": min_degree},
                    )

                else:
                    # Main query uses partial matching
                    main_query = """
                    MATCH (start)
                    WHERE
                        CASE
                            WHEN $inclusive THEN start.entity_id CONTAINS $entity_id
                            ELSE start.entity_id = $entity_id
                        END
                    WITH start
                    CALL apoc.path.subgraphAll(start, {
                        relationshipFilter: '',
                        minLevel: 0,
                        maxLevel: $max_depth,
                        bfs: true
                    })
                    YIELD nodes, relationships
                    WITH start, nodes, relationships
                    UNWIND nodes AS node
                    OPTIONAL MATCH (node)-[r]-()
                    WITH node, COALESCE(count(r), 0) AS degree, start, nodes, relationships
                    WHERE node = start OR EXISTS((start)--(node)) OR degree >= $min_degree
                    ORDER BY
                        CASE
                            WHEN node = start THEN 3
                            WHEN EXISTS((start)--(node)) THEN 2
                            ELSE 1
                        END DESC,
                        degree DESC
                    LIMIT $max_nodes
                    WITH collect({node: node}) AS filtered_nodes
                    UNWIND filtered_nodes AS node_info
                    WITH collect(node_info.node) AS kept_nodes, filtered_nodes
                    OPTIONAL MATCH (a)-[r]-(b)
                    WHERE a IN kept_nodes AND b IN kept_nodes
                    RETURN filtered_nodes AS node_info,
                            collect(DISTINCT r) AS relationships
                    """
                    result_set = await session.run(
                        main_query,
                        {
                            "max_nodes": MAX_GRAPH_NODES,
                            "entity_id": node_label,
                            "inclusive": inclusive,
                            "max_depth": max_depth,
                            "min_degree": min_degree,
                        },
                    )

                try:
                    record = await result_set.single()

                    if record:
                        # Handle nodes (compatible with multi-label cases)
                        for node_info in record["node_info"]:
                            node = node_info["node"]
                            node_id = node.id
                            if node_id not in seen_nodes:
                                result.nodes.append(
                                    KnowledgeGraphNode(
                                        id=f"{node_id}",
                                        labels=[node.get("entity_id")],
                                        properties=dict(node),
                                    )
                                )
                                seen_nodes.add(node_id)

                        # Handle relationships (including direction information)
                        for rel in record["relationships"]:
                            edge_id = rel.id
                            if edge_id not in seen_edges:
                                start = rel.start_node
                                end = rel.end_node
                                result.edges.append(
                                    KnowledgeGraphEdge(
                                        id=f"{edge_id}",
                                        type=rel.type,
                                        source=f"{start.id}",
                                        target=f"{end.id}",
                                        properties=dict(rel),
                                    )
                                )
                                seen_edges.add(edge_id)

                        logger.info(
                            f"Process {os.getpid()} graph query return: {len(result.nodes)} nodes, {len(result.edges)} edges"
                        )
                finally:
                    await result_set.consume()  # Ensure result set is consumed

            except neo4jExceptions.ClientError as e:
                logger.warning(f"APOC plugin error: {str(e)}")
                if node_label != "*":
                    logger.warning(
                        "Neo4j: falling back to basic Cypher recursive search..."
                    )
                    if inclusive:
                        logger.warning(
                            "Neo4j: inclusive search mode is not supported in recursive query, using exact matching"
                        )
                    return await self._robust_fallback(
                        node_label, max_depth, min_degree
                    )

        return result

    @with_semaphore
    async def _robust_fallback(
        self, node_label: str, max_depth: int, min_degree: int = 0
    ) -> KnowledgeGraph:
        """
        Fallback implementation when APOC plugin is not available or incompatible.
        This method implements the same functionality as get_knowledge_graph but uses
        only basic Cypher queries and recursive traversal instead of APOC procedures.
        """
        result = KnowledgeGraph()
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()

        async def traverse(
            node: KnowledgeGraphNode,
            edge: Optional[KnowledgeGraphEdge],
            current_depth: int,
        ):
            # Check traversal limits
            if current_depth > max_depth:
                logger.debug(f"Reached max depth: {max_depth}")
                return
            if len(visited_nodes) >= MAX_GRAPH_NODES:
                logger.debug(f"Reached max nodes limit: {MAX_GRAPH_NODES}")
                return

            # Check if node already visited
            if node.id in visited_nodes:
                return

            # Get all edges and target nodes
            async with self._driver.session(
                database=self._DATABASE, default_access_mode="READ"
            ) as session:
                query = """
                MATCH (a:base {entity_id: $entity_id})-[r]-(b)
                WITH r, b, id(r) as edge_id, id(b) as target_id
                RETURN r, b, edge_id, target_id
                """
                results = await session.run(query, entity_id=node.id)

                # Get all records and release database connection
                records = await results.fetch(
                    1000
                )  # Max neighbour nodes we can handled
                await results.consume()  # Ensure results are consumed

                # Nodes not connected to start node need to check degree
                if current_depth > 1 and len(records) < min_degree:
                    return

                # Add current node to result
                result.nodes.append(node)
                visited_nodes.add(node.id)

                # Add edge to result if it exists and not already added
                if edge and edge.id not in visited_edges:
                    result.edges.append(edge)
                    visited_edges.add(edge.id)

                # Prepare nodes and edges for recursive processing
                nodes_to_process = []
                for record in records:
                    rel = record["r"]
                    edge_id = str(record["edge_id"])
                    if edge_id not in visited_edges:
                        b_node = record["b"]
                        target_id = b_node.get("entity_id")

                        if target_id:  # Only process if target node has entity_id
                            # Create KnowledgeGraphNode for target
                            target_node = KnowledgeGraphNode(
                                id=f"{target_id}",
                                labels=list(f"{target_id}"),
                                properties=dict(b_node.properties),
                            )

                            # Create KnowledgeGraphEdge
                            target_edge = KnowledgeGraphEdge(
                                id=f"{edge_id}",
                                type=rel.type,
                                source=f"{node.id}",
                                target=f"{target_id}",
                                properties=dict(rel),
                            )

                            nodes_to_process.append((target_node, target_edge))
                        else:
                            logger.warning(
                                f"Skipping edge {edge_id} due to missing labels on target node"
                            )

                # Process nodes after releasing database connection
                for target_node, target_edge in nodes_to_process:
                    await traverse(target_node, target_edge, current_depth + 1)

        # Get the starting node's data
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            query = """
            MATCH (n:base {entity_id: $entity_id})
            RETURN id(n) as node_id, n
            """
            node_result = await session.run(query, entity_id=node_label)
            try:
                node_record = await node_result.single()
                if not node_record:
                    return result

                # Create initial KnowledgeGraphNode
                start_node = KnowledgeGraphNode(
                    id=f"{node_record['n'].get('entity_id')}",
                    labels=list(f"{node_record['n'].get('entity_id')}"),
                    properties=dict(node_record["n"].properties),
                )
            finally:
                await node_result.consume()  # Ensure results are consumed

            # Start traversal with the initial node
            await traverse(start_node, None, 0)

        return result

    @with_semaphore
    async def get_all_labels(self) -> list[str]:
        """
        Get all existing node labels in the database
        Returns:
            ["Person", "Company", ...]  # Alphabetically sorted label list
        """
        async with self._driver.session(
            database=self._DATABASE, default_access_mode="READ"
        ) as session:
            # Method 1: Direct metadata query (Available for Neo4j 4.3+)
            # query = "CALL db.labels() YIELD label RETURN label"

            # Method 2: Query compatible with older versions
            query = """
            MATCH (n)
            WHERE n.entity_id IS NOT NULL
            RETURN DISTINCT n.entity_id AS label
            ORDER BY label
            """
            result = await session.run(query)
            labels = []
            try:
                async for record in result:
                    labels.append(record["label"])
            finally:
                await (
                    result.consume()
                )  # Ensure results are consumed even if processing fails
            return labels

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    @with_semaphore
    async def delete_node(self, node_id: str) -> None:
        """Delete a node with the specified label

        Args:
            node_id: The label of the node to delete
        """

        async def _do_delete(tx: AsyncManagedTransaction):
            query = """
            MATCH (n:base {entity_id: $entity_id})
            DETACH DELETE n
            """
            result = await tx.run(query, entity_id=node_id)
            logger.debug(f"Deleted node with label '{node_id}'")
            await result.consume()  # Ensure result is fully consumed

        try:
            async with self._driver.session(database=self._DATABASE) as session:
                await session.execute_write(_do_delete)
        except Exception as e:
            logger.error(f"Error during node deletion: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    @with_semaphore
    async def remove_nodes(self, nodes: list[str]):
        """Delete multiple nodes

        Args:
            nodes: List of node labels to be deleted
        """
        for node in nodes:
            await self.delete_node(node)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(
            (
                neo4jExceptions.ServiceUnavailable,
                neo4jExceptions.TransientError,
                neo4jExceptions.WriteServiceUnavailable,
                neo4jExceptions.ClientError,
            )
        ),
    )
    @with_semaphore
    async def remove_edges(self, edges: list[tuple[str, str]]):
        """Delete multiple edges

        Args:
            edges: List of edges to be deleted, each edge is a (source, target) tuple
        """
        for source, target in edges:

            async def _do_delete_edge(tx: AsyncManagedTransaction):
                query = """
                MATCH (source:base {entity_id: $source_entity_id})-[r]-(target:base {entity_id: $target_entity_id})
                DELETE r
                """
                result = await tx.run(
                    query, source_entity_id=source, target_entity_id=target
                )
                logger.debug(f"Deleted edge from '{source}' to '{target}'")
                await result.consume()  # Ensure result is fully consumed

            try:
                async with self._driver.session(database=self._DATABASE) as session:
                    await session.execute_write(_do_delete_edge)
            except Exception as e:
                logger.error(f"Error during edge deletion: {str(e)}")
                raise

    @with_semaphore
    async def embed_nodes(
        self, algorithm: str
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        raise NotImplementedError

    def _worker_done_callback(self, future: asyncio.Future):
        """Callback function for worker task completion"""
        try:
            # Get worker ID from task name if available
            task_name = future.get_name() if hasattr(future, "get_name") else "Unknown"
            worker_id = (
                task_name.replace("neo4j_worker_", "")
                if task_name.startswith("neo4j_worker_")
                else "Unknown"
            )

            # Check task result
            future.result()
            logger.debug(f"Worker {worker_id} task completed normally")
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} task was cancelled")
        except Exception as e:
            logger.error(f"Worker {worker_id} task raised an exception: {str(e)}")
            # Log stack trace for debugging
            import traceback

            logger.debug(
                f"Worker {worker_id} exception traceback: {traceback.format_exc()}"
            )

    async def start_workers(self):
        """Start all worker tasks"""
        if self._workers_started:
            logger.warning("Neo4j workers are already started")
            return

        logger.info(f"Starting {self.NEO4J_WORKER_COUNT} Neo4j worker tasks")
        logger.debug(
            f"Worker configuration: max_concurrent_requests={self.NEO4J_MAX_CONCURRENT_REQUESTS}, "
            f"queue_size={self.NEO4J_QUEUE_SIZE}, operation_timeout={self.NEO4J_OPERATION_TIMEOUT}s, "
            f"long_tx_threshold={self.LONG_TRANSACTION_THRESHOLD}s"
        )

        for i in range(self.NEO4J_WORKER_COUNT):
            logger.debug(f"Creating worker {i}")
            worker = asyncio.create_task(
                neo4j_worker(i, self._task_queue, self._semaphore),
                name=f"neo4j_worker_{i}",  # Set task name for better identification
            )
            worker.add_done_callback(self._worker_done_callback)
            self._workers.append(worker)
            logger.debug(f"Worker {i} created and added to worker pool")

        self._workers_started = True
        logger.info(f"All {len(self._workers)} Neo4j workers started successfully")

    async def stop_workers(self):
        """Stop all worker tasks"""
        # If workers haven't been started, nothing to do
        if not self._workers_started:
            logger.info("Neo4j workers were not started, nothing to stop")
            return

        logger.info(f"Stopping {len(self._workers)} Neo4j worker tasks")

        # Check if there are any tasks still in the queue
        remaining_tasks = self._task_queue.qsize()
        if remaining_tasks > 0:
            logger.warning(
                f"There are still {remaining_tasks} tasks in the queue that will not be processed"
            )

        # Cancel all workers
        for i, worker in enumerate(self._workers):
            if not worker.done():
                logger.debug(f"Cancelling worker {i}")
                worker.cancel()
            else:
                logger.debug(f"Worker {i} is already done, no need to cancel")

        # Wait for all workers to finish
        if self._workers:
            logger.debug(f"Waiting for {len(self._workers)} workers to finish")
            start_time = time.time()
            await asyncio.gather(*self._workers, return_exceptions=True)
            duration = time.time() - start_time
            logger.debug(f"All workers finished in {duration:.3f}s")

        # Reset workers state
        worker_count = len(self._workers)
        self._workers = []
        self._workers_started = False

        logger.info(f"All {worker_count} Neo4j worker tasks stopped successfully")

        # Log queue state after stopping workers
        logger.debug(
            f"Queue state after stopping workers: size={self._task_queue.qsize()}/{self._task_queue.maxsize}"
        )
