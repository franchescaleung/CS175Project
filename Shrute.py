from schrutepy import schrutepy


if __name__ == '__main__':

	df = schrutepy.load_schrute()

	# print(df.head)

	# for i in range(len(df.character)):
	# 	if df.character[i] == "Dwight":
	# 		print(df.text[i])

	print(df.text[1])
	print(df.text_w_direction[1])