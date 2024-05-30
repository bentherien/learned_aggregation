
import jax
import gc
from helpers import get_mup_lrs_from_state
import threading

class MuTask(object):
  def get_mup_state(self,state):
    if self.mup_state is None:
      device = jax.devices()[0]
      self.mup_state = get_mup_lrs_from_state(state)
      self.mup_state = jax.tree_map(lambda x: jax.device_put(x, device), self.mup_state)

    state['mup_lrs_to_use'] = self.mup_state
    return state
  
  def init_mup_state(self): 
    #create and save mup state outside of jit
    key = jax.random.PRNGKey(0)
    params, state = self.init_with_state(key)
    del params
    del state
        
    # Force garbage collection in a separate thread to make it non-blocking
    gc_thread = threading.Thread(target=gc.collect)
    gc_thread.start()
    # gc_thread.join()  # Optionally wait for the GC to complete