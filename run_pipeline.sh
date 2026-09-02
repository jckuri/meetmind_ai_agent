rm -rf executions/
mkdir executions

python -u sft.py 2>&1 | tee executions/sft.py.txt
python -u agentic_traces.py 2>&1 | tee executions/agentic_traces.py.txt
python -u agentic_rlft.py 2>&1 | tee executions/agentic_rlft.py.txt
