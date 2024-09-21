import configparser

def get_list(option):
	return list(map(float, ''.join(option.split()).split(',')))

def get_list_int(option):
	return list(map(int, ''.join(option.split()).split(',')))

def get_matrix(option):
	matrix = []
	for i in ''.join(option.split()).split(';'):
		matrix.append(list(map(int,i.split(','))))
	return matrix

def read_config(filename):

	config = configparser.ConfigParser()
	config.read(filename)

	intp_mim = config.getint('decaps', 'intp_mim')
	chip_mos = config.getint('decaps', 'chip_mos')
	NCAP = config.getint('decaps', 'NCAP')
	if intp_mim == 121:
		intp_n = []
	else:
		intp_n = get_list_int(config.get('decaps', 'intp_n'))
	chip_n = get_list_int(config.get('decaps', 'chip_n'))

	return intp_mim, chip_mos, NCAP, intp_n, chip_n

