import scipy.ndimage as ndi
ndi.morphology = ndi  # scipy récent n'a plus ndimage.morphology, utilisé par synthstrip
from nipreps.synthstrip.cli import main
main()
