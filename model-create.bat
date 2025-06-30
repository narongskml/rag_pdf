REM ===== Pull base model ====
ollama pull qwen2.5vl:3b
ollama pull llama3.2
ollama pull gemma3:1b
ollama pull deepseek-r1:1.5b

REM ===== Remove Model =====
ollama rm pdf-llama
ollama rm pdf-deepseek
ollama rm pdf-gemma
ollama rm pdf-qwen 

REM ===== Create Model =====
ollama create pdf-llama -f Modelfile-llama3.2
ollama create pdf-deepseek -f Modelfile-deepseek
ollama create pdf-gemma -f Modelfile-gemma
ollama create pdf-qwen -f Modelfile-qwen