import pygame

pygame.mixer.init()

def playmp3(filename):
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy() == True:
        continue

playmp3("kripya dhyan deejiye.mp3")
playmp3("aapke saamne nukili cheez hai.mp3")
playmp3("aapko chot lag sakti hai.mp3")

playmp3("aapke saamne.mp3")
playmp3("chaar.mp3")
playmp3("log hain.mp3")

playmp3("there are.mp3")
playmp3("more than four.mp3")
playmp3("people in front of you.mp3")

playmp3("aapke saamne.mp3")
playmp3("eiffel tower.mp3")
playmp3("hai.mp3")

playmp3("aapke saamne.mp3")
playmp3("modi.mp3")
playmp3("hai.mp3")

playmp3("aapke saamne.mp3")
playmp3("neela.mp3")
playmp3("santra.mp3")
playmp3("hai.mp3")

playmp3("kripya dhyan deejiye.mp3")
playmp3("aapke saamne nukili cheez hai.mp3")
playmp3("aapko chot lag sakti hai.mp3")
