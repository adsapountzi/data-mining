#Authors : Koukougiannis Dimitris 2537
#		   Sapountzi Athanasia Despooina 2624

#this program implements the gui of the our application, by using the appropriate widgets from the tkinter library.

import tkinter as tk
import PIL
from tkinter import *
from PIL import ImageTk, Image
import diabetes
import pickle
fields = 'Pregnancies', 'Glucose', 'BloodPressure (mm Hg)', 'SkinThickness (mm)', 'Insulin (mu U/ml)', 'BMI', 'Diabetes Pedigree', 'Age'

def fetch(entries, root):
	text = []
	
	for entry in entries:
		field = entry[0]
		text.append(entry[1].get())
	#predict the outcome
	output = diabetes.check_input({"A": text[0], "B": text[1], "C": text[2], "D": text[3], "E": text[4], "F": text[5], "g": text[6], "H": text[7]})
	
	#create text widget for text outcome.
	text = Text(root, height = 5, width = 40, bg = 'royalBlue2', fg = 'white')
	#position text outcome. 
	text.place(x = 20,y = 390)

	l = Label(root, text = "RESULTS", fg = 'RoyalBlue2')
	l.config(font =("Courier", 14))
	l.place(x = 140,y = 360)

	#check the result of prediction and display the appropriate text outcome.

	if(output == 1) :	
		text.insert(INSERT, "Patient has diabetes!")
	else :
		text.insert(INSERT, "Patient does not have diabetes!")
def makeform(root, fields):
	entries = []
	for count, field in enumerate(fields):
		name_label = tk.Label(root, text = field, font=('calibre',10, 'bold'), fg='midnight blue')
  
		# creating an entry widget
		name_entry = tk.Entry(root, font=('calibre',10,'normal'))
		
		#position the entry fields in gui
		name_label.grid(row=count,column=0, pady = 10)

		name_entry.grid(row=count,column=2, pady = 10)  		
		entries.append((field, name_entry))
	return entries

if __name__ == '__main__':
	diabetes.train()
	root = tk.Tk()
	root.title("Diabetes Prediction Application")
	root.geometry("820x650")
	
	#insert backround image of gui.	  
	img = ImageTk.PhotoImage(Image.open('med.png'), master = root)

	label = Label( root, image=img)
	label.place(x=200, y=0)
	ents = makeform(root, fields)
	root.bind('<Return>', (lambda event, e=ents: fetch(e,root)))

	#create button submit with button widget
	b1 = tk.Button(root, text='Submit',  bg='midnight blue', fg='white',
	 			  command=(lambda e=ents: fetch(e, root)))
	#position button Submit
	b1.grid(row = 9, column = 0, pady = 200)


	#create button quit with button widget
	b2 = tk.Button(root, text='Quit', bg= 'red3',  fg='white', command=root.quit)

	#position button quit
	b2.grid(row = 9, column = 1, pady = 200)

	root.mainloop()
