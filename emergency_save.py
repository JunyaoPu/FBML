import pickle
import sys

# Debug: prove we're running
open('/Users/jakob/Development/Games/FBML/models/INJECT_WORKED.txt', 'w').write('injection executed\n')

# Try to find population in the __main__ module's globals
main_mod = sys.modules.get('__main__')
if main_mod:
    pop = getattr(main_mod, 'population', None)
    if pop:
        pickle.dump([b.tensors for b in pop.individuals], open('/Users/jakob/Development/Games/FBML/models/EMERGENCY.pkl', 'wb'))
        open('/Users/jakob/Development/Games/FBML/models/SAVE_SUCCESS.txt', 'w').write(f'SAVED {len(pop.individuals)} birds, best_ever_dist={pop.best_ever_distance}')
    else:
        # List all attrs in main
        attrs = [a for a in dir(main_mod) if not a.startswith('_')]
        open('/Users/jakob/Development/Games/FBML/models/MAIN_ATTRS.txt', 'w').write('\n'.join(attrs))
else:
    open('/Users/jakob/Development/Games/FBML/models/NO_MAIN.txt', 'w').write('no __main__')
