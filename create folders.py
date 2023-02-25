import os

path = "naruto handsign data/validation/"

signlist = ["bird","boar","dog","dragon","hare","horse","monkey","ox"]

for sign in signlist:
    try:
        os.mkdir(path + sign)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)