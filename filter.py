
f = open ("Dwight.txt", 'r')
g = open ("Filtered.txt", 'w')
prev_line = 14
prev_speaker = ""
numbers = []
for line in f:
	temp = line.split()
	x = int(temp[0])
	if prev_line != x-1: #we jumped to a new section
		if prev_speaker != "Dwight": # and the previous conversation did not end with Dwight speaking
			numbers.pop()
	numbers.append(x) #add it to our list of desires entries
	prev_line = x #update
	prev_speaker = temp[1]
f.close()
f = open("Dwight.txt", 'r')
x = numbers[0]
for line in f:
	temp = line.split()
	y = int(temp[0])
	if y == x:
		g.write(line)
		break
	else:
		continue
for n in range(1,len(numbers)):
	if numbers[n]-x > 1:
		g.write("\n")
	for line in f:
		temp = line.split()
		y = int(temp[0])
		if y == numbers[n]:
			g.write(line)
			x = y
			break
		else:
			continue


