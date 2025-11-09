# Case Closed Agent Template

### Explanation of Files

This template provides a few key files to get you started. Here's what each one does:

#### `agent.py`
**This is the most important file. This is your starter code, where you will write your agent's logic.**

*   DO NOT RENAME THIS FILE! Our pipeline will only recognize your agent as `agent.py`.
*   It contains a fully functional, Flask-based web server that is already compatible with the Judge Engine's API.
*   It has all the required endpoints (`/`, `/send-state`, `/send-move`, `/end`). You do not need to change the structure of these.
*   Look for the `send_move` function. Inside, you will find a section marked with comments: `# --- YOUR CODE GOES HERE ---`. This is where you should add your code to decide which move to make based on the current game state.
*   Your agent can return moves in the format `"DIRECTION"` (e.g., `"UP"`, `"DOWN"`, `"LEFT"`, `"RIGHT"`) or `"DIRECTION:BOOST"` (e.g., `"UP:BOOST"`) to use a speed boost.

#### `requirements.txt`
**This file lists your agent's Python dependencies.**

*   Don't rename this file either.
*   It comes pre-populated with `Flask` and `requests`.
*   If your agent's logic requires other libraries (like `numpy`, `scipy`, or any other package from PyPI), you **must** add them to this file.
*   When you submit, our build pipeline will run `pip install -r requirements.txt` to install these libraries for your agent.

#### `judge_engine.py`
**A copy of the runner of matches.**

*   The judge engine is the heart of a match in Case Closed. It can be used to simulate a match.
*   The judge engine can be run only when two agents are running on ports `5008` and `5009`.
*   We provide a sample agent that can be used to train your agent and evaluate its performance.

#### `case_closed_game.py`
**A copy of the official game state logic.**

*   Don't rename this file either.
*   This file contains the complete state of the match played, including the `Game`, `GameBoard`, and `Agent` classes.
*   While your agent will receive the game state as a JSON object, you can read this file to understand the exact mechanics of the game: how collisions are detected, how trails work, how boosts function, and what ends a match. This is the "source of truth" for the game rules.
*   Key mechanics:
    - Agents leave permanent trails behind them
    - Hitting any trail (including your own) causes death
    - Head-on collisions: draw (both agents die)
    - Each agent has 3 speed boosts (moves twice instead of once)
    - The board has torus (wraparound) topology
    - Game ends after 200 turns or when one/both agents die

#### `sample_agent.py`
**A simple agent that you can play against.**

*   The sample agent is provided to help you evaluate your own agent's performance. 
*   In conjunction with `judge_engine.py`, you should be able to simulate a match against this agent.

#### `local-tester.py`
**A local tester to verify your agent's API compliance.**

*   This script tests whether your agent correctly implements all required endpoints.
*   Run this to ensure your agent can communicate with the judge engine before submitting.

#### `Dockerfile`
**A copy of the Dockerfile your agent will be containerized with.**

*   This is a copy of a Dockerfile. This same Dockerfile will be used to containerize your agent so we can run it on our evaluation platform.
*   It is **HIGHLY** recommended that you try Dockerizing your agent once you're done. We can't run your agent if it can't be containerized.
*   There are a lot of resources at your disposal to help you with this. We recommend you recruit a teammate that doesn't run Windows for this. 

#### `.dockerignore`
**A .dockerignore file doesn't include its contents into the Docker image**

*   This `.dockerignore` file will be useful for ensuring unwanted files do not get bundled in your Docker image.
*   You have a 5GB image size restriction, so you are given this file to help reduce image size and avoid unnecessary files in the image.

#### `.gitignore`
*   A standard configuration file that tells Git which files and folders (like the `venv` virtual environment directory) to ignore. You shouldn't need to change this.


### Testing your agent:
**Both `agent.py` and `sample_agent.py` come ready to run out of the box!**

*   To test your agent, you will likely need to create a `venv`. Look up how to do this. 
*   Next, you'll need to `pip install` any required libraries. `Flask` is one of these.
*   Finally, in separate terminals, run both `agent.py` and `sample_agent.py`, and only then can you run `judge_engine.py`.
*   You can also run `local-tester.py` to verify your agent's API compliance before testing against another agent.


---

## Training Your Agent with PPO

This repository includes a full PPO (Proximal Policy Optimization) training system for developing Tron agents through self-play.

### Quick Start

**Run a smoke test (128 steps):**
```bash
python -m training.trainer --max-steps 128 --rollout-length 64 --batch-size 32 --epochs 2
```

**Run full training (200k steps):**
```bash
bash scripts/run_training.sh
```

**Custom training run:**
```bash
python -m training.trainer \
  --seed 42 \
  --max-steps 200000 \
  --rollout-length 2048 \
  --batch-size 256 \
  --epochs 10 \
  --lr 0.0003 \
  --device cpu \
  --use-frozen-opponent true \
  --opponent-update-interval 5
```

**Resume from checkpoint:**
```bash
python -m training.trainer --ckpt-path training/checkpoints/checkpoint_step_50000.pt
```

### Training in Docker

```bash
# Build the training image
docker build -t tron-ppo:cpu .

# Run training (mounts current directory for checkpoints/logs)
docker run --rm -it -v "$PWD:/app" tron-ppo:cpu

# Or run with custom parameters
docker run --rm -it -v "$PWD:/app" tron-ppo:cpu \
  python -m training.trainer --max-steps 100000
```

### Expected Outputs

**Console:** Update metrics printed every ~4096 steps showing:
- Episode statistics (win rate, avg reward, avg length)
- Training metrics (policy loss, value loss, entropy, KL divergence)
- Timing information (rollout time, update time)

**CSV Log:** `runs/<RUN_NAME>/metrics.csv` contains all metrics with headers:
```
step,update,episodes,avg_reward,avg_len,win_rate,draw_rate,
policy_loss,value_loss,entropy,clip_frac,approx_kl,rollout_sec,update_sec
```

**Checkpoints:** Saved to `training/checkpoints/` at intervals (default: every 50k steps):
- `checkpoint_step_<N>.pt` - Periodic checkpoints
- `final_checkpoint.pt` - Saved at training completion

### Configuration Parameters

All training parameters can be configured via CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--seed` | 42 | Random seed for reproducibility |
| `--max-steps` | 200000 | Maximum training steps |
| `--device` | cpu | Device (CPU only per competition rules) |
| `--rollout-length` | 2048 | Steps per rollout |
| `--batch-size` | 256 | Mini-batch size |
| `--epochs` | 10 | PPO epochs per update |
| `--lr` | 0.0003 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--max-kl` | 0.03 | KL divergence threshold for early stopping |
| `--use-frozen-opponent` | true | Use frozen opponent for stable self-play |
| `--opponent-update-interval` | 5 | Update opponent every N updates |
| `--csv-log` | true | Enable CSV logging |
| `--tensorboard` | false | Enable TensorBoard logging (optional) |

### Performance Metrics

**Model Specifications:**
- Architecture: 2-layer MLP encoder + actor/critic heads
- Model size: <5MB (enforced)
- Forward pass: <5ms per step on typical CPU
- Board size: 15x15 (configurable)

**Training Performance:**
- Rollout collection: ~2-5 seconds per 2048 steps
- PPO update: ~5-10 seconds per update
- Expected win rate improvement: ≥5pp over first 50k steps vs. initial frozen opponent

### Testing

Run the test suite to verify installation:

```bash
# Install test dependencies
pip install pytest

# Run all tests
pytest -v

# Run only unit tests
pytest -v -m unit

# Run only integration tests (slower)
pytest -v -m integration

# Run specific test file
pytest tests/test_policy.py -v
```

### Troubleshooting

**Issue:** `ModuleNotFoundError: No module named 'torch'`
- **Solution:** Install dependencies: `pip install -r requirements.txt`

**Issue:** Training shows NaN losses
- **Solution:** This is automatically handled with NaN guards. Check logs for "skipped updates" counter.

**Issue:** Win rate not improving
- **Solution:** 
  - Try lowering learning rate (`--lr 0.0001`)
  - Increase entropy coefficient for more exploration (`--entropy-coeff 0.02`)
  - Check that opponent is updating at reasonable intervals

**Issue:** Out of memory
- **Solution:** Reduce batch size (`--batch-size 128`) or rollout length (`--rollout-length 1024`)


---

### Disclaimers:
* There is a 5GB limit on Docker image size, to keep competition fair and timely.
* Due to platform and build-time constraints, participants are limited to **CPU-only PyTorch**; GPU-enabled versions, including CUDA builds, are disallowed. Any other heavy-duty GPU or large ML frameworks (like Tensorflow, JAX) will not be allowed.
* Ensure your agent's `requirements.txt` is complete before pushing changes.
* If you run into any issues, take a look at your own agent first before asking for help.
