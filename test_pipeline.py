import asyncio
from backend.pipeline.orchestrator import run_pipeline

async def main():
    result = await run_pipeline(client_id="test_user_123")

    print("\n=== PIPELINE RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())