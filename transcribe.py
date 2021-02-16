from schrutepy import schrutepy


df = schrutepy.load_schrute()
dSpeak = []
for i in range(1, len(df.character)):
	if (df.character[i] == "Dwight" and df.character[i-1]):
		print(df.text[i])
		

	

