# Reinforcement-for-Label-Correction
run_log=K:\rlnlc\outputs\mnist_speed_rtx5080\run.log
Traceback (most recent call last):
  File "k:\rlnlc\mnist_speed_test.py", line 944, in <module>
    run_with_file_logging()
  File "k:\rlnlc\mnist_speed_test.py", line 940, in run_with_file_logging
    main()
  File "k:\rlnlc\mnist_speed_test.py", line 586, in main
    torch.cuda.reset_peak_memory_stats(device)
  File "C:\Users\cream\anaconda3\Lib\site-packages\torch\cuda\memory.py", line 376, in reset_peak_memory_stats
    return torch._C._cuda_resetPeakMemoryStats(device)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Invalid device argument 
