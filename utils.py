from datetime import datetime

def infoPrint(str):
	diff = datetime.now() - infoPrint.startTime
	print("\033[92m[INFO] \033[94m{0:>5} \033[0m{1}".format(diff.seconds, str))
