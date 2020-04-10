import pygame
import constants


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.image = pygame.image.load(image_file)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location

    def scrollToleft(self):
        self.rect.left += 3 * constants.WINDOW_WIDTH if self.rect.left <= - constants.WINDOW_WIDTH else - constants.SPEED