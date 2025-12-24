print("Testing import conflict with mplsoccer...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print("Matplotlib imported")
    
    from mplsoccer import Pitch
    print("mplsoccer imported")
    
    import tls_client
    print("tls_client imported")
    
    import pandas as pd
    print("pandas imported")
except Exception as e:
    print(f"Error: {e}")
print("Done")
