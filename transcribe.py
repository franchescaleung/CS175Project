from schrutepy import schrutepy

f = open("Dwight.txt",'w')
g = open("query.txt",'w')
df = schrutepy.load_schrute()
dSpeak = []
for i in range(len(df.character)):
	if df.character[i] == "Dwight":
		f.write(str(i) + ". ")
		f.write(str(df.text[i]))
		f.write('\n')
		if df.character[i-1] != "Dwight":
			g.write(str(i-1) + ". ")	
			g.write(str(df.text[i-1]))
			g.write('\n')
f.close()
g.close()
	

