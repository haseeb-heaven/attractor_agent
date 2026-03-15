import httpx
import json
import time

def test_api():
    base_url = "http://localhost:8000/api/v1"
    
    # 1. Create run
    print("Creating run...")
    resp = httpx.post(f"{base_url}/runs", json={
        "prompt": "Test pipeline with human step",
        "language": "Python"
    })
    run_id = resp.json()["run_id"]
    print(f"Run ID: {run_id}")
    
    # 2. Check events stream (SSE)
    print("Checking events...")
    with httpx.stream("GET", f"{base_url}/runs/{run_id}/events") as response:
        for line in response.iter_lines():
            if line.startswith("data:"):
                event = json.loads(line[6:])
                print(f"Event: {event['event_kind']}")
                if event['event_kind'] == "PIPELINE_STARTED":
                    break
    
    # 3. Check status
    resp = httpx.get(f"{base_url}/runs/{run_id}")
    print(f"Status: {resp.json()['status']}")
    
    # 4. Check questions
    print("Checking questions...")
    resp = httpx.get(f"{base_url}/runs/{run_id}/questions")
    print(f"Questions: {resp.json()}")

if __name__ == "__main__":
    try:
        test_api()
    except Exception as e:
        print(f"Test failed: {e}")
