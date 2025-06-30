REM ===== Pull base model ====
ollama pull qwen2.5vl:3b
ollama pull llama3.2
ollama pull gemma3:1b
ollama pull phi3:latest

REM ===== Remove Model =====
ollama rm pdf-llama
ollama rm pdf-gemma
ollama rm pdf-qwen 
ollama rm pdf-phi3

REM ===== Create Model =====
ollama create pdf-llama -f Modelfile-llama3.2
ollama create pdf-gemma -f Modelfile-gemma
ollama create pdf-qwen -f Modelfile-qwen
ollama create pdf-phi3 -f Modelfile-phi3