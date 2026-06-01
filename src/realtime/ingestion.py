import asyncio
import json
import logging
from typing import Dict, Any

logger = logging.getLogger("ingestion_server")

class IngestionServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9999, max_queue_size: int = 50):
        self.host = host
        self.port = port
        self.max_queue_size = max_queue_size
        self.queue = asyncio.Queue(maxsize=max_queue_size)
        self.server = None
        self.active_connections = 0
        self.total_flows_received = 0
        self.is_running = False

    async def start(self):
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        self.is_running = True
        logger.info(f"Ingestion TCP server listening on {self.host}:{self.port} (Max Queue: {self.max_queue_size})")
        # Run server in the background
        asyncio.create_task(self.server.serve_forever())

    async def stop(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.is_running = False
            logger.info("Ingestion TCP server stopped.")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.active_connections += 1
        addr = writer.get_extra_info('peername')
        logger.info(f"Accepted flow stream connection from {addr}")

        buffer = ""
        try:
            while self.is_running:
                data = await reader.read(4096)
                if not data:
                    logger.info(f"Connection from {addr} closed by client.")
                    break
                
                buffer += data.decode(errors='ignore')
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        flow = json.loads(line)
                        self.total_flows_received += 1
                        
                        # backpressure-aware insertion: blocks if queue is full
                        # this blocks the client reader task, leading to TCP socket buffer filling,
                        # which triggers backpressure at the sender.
                        if self.queue.full():
                            logger.warning(f"Ingestion queue full ({self.queue.qsize()}/{self.max_queue_size}). Backpressuring sender...")
                        
                        await self.queue.put(flow)
                    except json.JSONDecodeError:
                        logger.error(f"Malformed JSON received: {line[:100]}...")
                    except Exception as e:
                        logger.error(f"Error putting flow in queue: {e}")
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error handling connection from {addr}: {e}", exc_info=True)
        finally:
            self.active_connections = max(0, self.active_connections - 1)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def get_next_flow(self) -> Dict[str, Any]:
        """Fetch the next flow from the queue. Blocks if empty."""
        return await self.queue.get()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_connections": self.active_connections,
            "total_flows_received": self.total_flows_received,
            "queue_size": self.queue.qsize(),
            "max_queue_size": self.max_queue_size,
            "queue_percent_full": (self.queue.qsize() / self.max_queue_size) * 100 if self.max_queue_size > 0 else 0
        }
