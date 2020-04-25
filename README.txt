In order to run both baseline models, do as follows:

For GRU model, train gru_model by running python gru_model.py.
Once this is trained, the model will be saved and you can run run_model.py with the parameters in the command line as follows:

run_model.py [input string (in quotes)] gru [number of characters to generate]
If these are not satisfied the program will tell you what you are missing.

To run lstm model, go to line 221 and change the input_text parameter to an input text string of your choosing.
Run python lstm_garbage_model.py and the model will train and then output the output_length variable number of additional characters (line 224).

To continue running additional strings in lstm_garbage_model, comment line 212 (model.fit()) and run the program again. This will make use of the saved checkpoints to generate new strings based on changed input strings.

Please take a look at enhanced_model.py to see comments as well as the direction of the proposed extended model. 
