import pygame
import constants

class Welcome(pygame.sprite.Sprite):
    def __init__(self, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load('assets/message.png')
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
