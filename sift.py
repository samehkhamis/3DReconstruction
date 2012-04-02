import os
from subprocess import *
from siftmatch import *

def detect(imfilename, keyfilename):
	dir, curfile = os.path.split(__file__)
	imfile = file(imfilename, 'rb')
	keyfile = file(keyfilename, 'wb')
	
	if os.name == 'nt':
		p = Popen([dir + os.path.sep + "siftWin32"], bufsize=1024, stdin=imfile, stdout=keyfile)
	else:
		p = Popen([dir + os.path.sep + "sift"], bufsize=1024, stdin=imfile, stdout=keyfile, close_fds=True)
	
	p.wait()
	