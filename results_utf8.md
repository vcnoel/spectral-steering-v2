python : Traceback (most recent call last):
Au caractère Ligne:1 : 1
+ python scripts/run_rebuttal_subsets.py > results.md 2>&1
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (Traceback (most  
   recent call last)::String) [], RemoteException
    + FullyQualifiedErrorId : NativeCommandError
 
  File "C:\Users\valno\Dev\spectral-steering-v2\scripts\run_re
buttal_subsets.py", line 59, in <module>
    main()
    ~~~~^^
  File "C:\Users\valno\Dev\spectral-steering-v2\scripts\run_re
buttal_subsets.py", line 22, in main
    print("\U0001f680 Starting Fast-Track Rebuttal 
Experiments...")
    ~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\valno\miniconda3\Lib\encodings\cp1252.py", 
line 19, in encode
    return 
codecs.charmap_encode(input,self.errors,encoding_table)[0]
           
~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character 
'\U0001f680' in position 0: character maps to <undefined>
