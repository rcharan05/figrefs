cd /home/charan/proj/figrefs/eval/figsum
JUDGE_PROMPT=v2 JUDGE_OUT=llm_q25_v2.json python3 llm_judge.py window 2>&1 | grep -v -i "warn\|Loading"
JUDGE_MODEL=Qwen/Qwen3-1.7B JUDGE_PROMPT=v2 JUDGE_OUT=llm_q3_v2.json python3 llm_judge.py window 2>&1 | grep -v -i "warn\|Loading"
