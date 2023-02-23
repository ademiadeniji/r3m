import os
# Iterate through files in directory realvideos
# and parse the video files to extract the frames
# and save them in a directory called frames
videos = os.listdir('/home/ademi_adeniji/realvideos')
# create a directory called frames if doesn't exist, if it does exist, get rid of it
if os.path.isdir('/home/ademi_adeniji/realvideos/frames'):
    os.system('rm -rf /home/ademi_adeniji/realvideos/frames')
os.mkdir('/home/ademi_adeniji/realvideos/frames')
for video in videos:
    # if directory already exists, delete it
    if os.path.isdir('/home/ademi_adeniji/realvideos/frames/' + video[0:-4]):
        os.system('rm -rf /home/ademi_adeniji/realvideos/frames/' + video[0:-4])
    os.mkdir('/home/ademi_adeniji/realvideos/frames/' + video[0:-4])
    # parse the video file and extract the frames
    # and save them in the directory created above
    # make sure frames are 0-indexed and size is 64x64
    os.system('ffmpeg -i /home/ademi_adeniji/realvideos/' + video + ' -vf scale=64:64 /home/ademi_adeniji/realvideos/frames/' + video[0:-4] + '/%d.jpg')

    # go through each frame and rename it to be 0-indexed
    frames = os.listdir('/home/ademi_adeniji/realvideos/frames/' + video[0:-4])
    for i in range(1, len(frames)+1, 1):
        os.rename('/home/ademi_adeniji/realvideos/frames/' + video[0:-4] + '/' + f'{i}.jpg', '/home/ademi_adeniji/realvideos/frames/' + video[0:-4] + '/' + str(i-1) + '.jpg')
    # get rid of fr folder
    os.system('rm -rf /home/ademi_adeniji/realvideos/frames/fr')