import easygui
import os

pgm="python modules/yoloface.py"
pgm2="python modules/encode_faces_yolo.py"
choice1=easygui.buttonbox('Select what you have to do',title='Select',choices=('Add Faces','Recognize faces'))

if choice1=='Add Faces':
	trainmode=easygui.choicebox('Select training input',title='Training',choices=('Image Folder','Append from Video File'))
	print('Trainmode:',trainmode)
	cc=easygui.ccbox(msg='For proceeding further you must have images of people grouped in respective folder with theri names, all grouped under a single folder. If you dont have the data please click cancel and return later.')
	if cc:
		if trainmode=='Image Folder':
			dirpath=easygui.diropenbox(title='Image directory')
			print(dirpath)
			encodingpath=easygui.filesavebox(msg='Save your face encodings',title='Face encodings',default='encodings/*.pickle')
			os.system(pgm2+' --dataset \''+dirpath+'\' --encodings \''+encodingpath+'\'')
		elif trainmode=='Append from Video File':
			if not os.path.isdir('tempdataset'):os.mkdir('tempdataset')
			vidpath=easygui.fileopenbox(title='Videofile')
			start=vidpath.rfind('/')+1
			end=vidpath.rfind('.')
			name=vidpath[start:end]
			encodingpath=easygui.filesavebox(msg='Append your face encodings to',title='Face encodings',default='encodings/*.pickle')
			rotate=easygui.choicebox(msg='Angle at which frame should be rotated',title='Rotation',choices=('90 degree clockwise','180 degree','90 degree anticlockwise'))
			if rotate in ('90 degree clockwise','180 degree','90 degree anticlockwise'):
				angle='0' if rotate=='90 degree clockwise' else '1' if rotate=="180 degree" else '2'
				print(rotate,angle)
				print('python modules/video2img.py --video \''+vidpath+'\' --name \''+name+'\' --rotate '+angle)
				os.system('python modules/video2img.py --video \''+vidpath+'\' --name \''+name+'\' --rotate 0')
			else:
				os.system('python modules/video2img.py --video \''+vidpath+'\' --name \''+name+'\'')
			os.system(pgm2+' --dataset tempdataset --encodings \''+encodingpath+'\'')			
			os.system('rm -rf tempdataset')
elif choice1=='Recognize faces':
	encoding=easygui.fileopenbox(default='encodings/*.pickle')
	if encoding:
		mode=easygui.buttonbox('Select input mode',title='Mode',choices=('Image','Video','Camera'))
		if mode=='Image':
			image=easygui.fileopenbox(msg='Select the image file to recognize faces',title='Select Image',filetypes=('*.jpg','*.png'))
			os.system(pgm+" --image \'"+image+"\' --encodings \'"+encoding+"\'")
		elif mode=='Video':
			video=easygui.fileopenbox(msg='Select the video file to recognize faces',title='Select Video',filetypes=('*.mp4','*.mkv','*.avi'))
			os.system(pgm+" --video \'"+video+"\' --encodings \'"+encoding+"\'")
		elif mode=='Camera':
			os.system(pgm+" --src 2"+" --encodings \'"+encoding+"\'") # video source is 2 for my laptop
	
