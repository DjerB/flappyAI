import pygame


class Play(pygame.sprite.Sprite):
    def __init__(self, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.transform.scale2x(pygame.image.load('assets/play.png'))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location
