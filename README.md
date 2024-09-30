1. Start Ollama service : service ollama start
2. Check that server is running : http://127.0.0.1:11434
3. Launch code to fectch pdf & ask questions : python adb_main.py file.pdf

Running on my small laptop (Core™ i7-7600U, 16 Go), here's the results, reading the Argo manual (2Mo, 114 pages)
```
(LLM-PDF) $ python adb_main.py 94819.pdf 
Fetching 9 files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 9/9 [00:00<00:00, 3999.65it/s]
PDF successfully ingested!
>> Enter your question : What is Argo ?
Answer:  Argo is a global program that operates and manages over 3000 floats distributed in all oceans to monitor ocean conditions. The data collected is managed by the Argo data management group and made available in the NetCDF format, which is widely accepted in the scientific community. Users are responsible for assessing data accuracy and should report any issues to Argo through their website or email.
>> Enter your question : What is the parameter's code for grounding ?
Answer:  The context provided does not contain information about a parameter for grounding with a numeric code. The documents discuss various configuration parameters and their definitions, as well as some telecommunication system details.
>> Enter your question : What is the code identifying the float type "Deep Arvor" ?
Answer:  The context does not provide specific information about a float type called "Deep Arvor." The documents mention an "Arvor/Provor float" type under the name "APMT," but they don't explicitly identify a "Deep Arvor" variant.
```
All this in ~ 20 minutes
