# PokemonEbayBot
--------------------
## How to run the agent
Download and run the agent

1. Download or clone this repo

2. Create a virtual environment

   ```sh
   python -m venv .venv
   ```

3. Activate the virtual environment:
- **Windows:** `.venv\Scripts\activate`
- **macOS/Linux:** `source .venv/bin/activate`

4. Install Dependencies
   ```sh
   pip install -r requirements.txt
   ```

5. Copy `.env.example` to `.env` and fill in your API keys.

6. Run the agent
   ```sh
   python -m my_agent.agent
   ```

## Run on a server
uv run langgraph dev

### Test API through command line
uv run test_client.py


## How to run tests

See [tests/README.md](tests/README.md)