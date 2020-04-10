import pygame
import constants


class Pipe(pygame.sprite.Sprite):
    count = 0

    def __init__(self, image_file, height, offset_x, offset_y=0, origin='top'):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        Pipe.count += 1
        self.image = pygame.image.load(image_file)
        if origin == 'top':
            self.image = pygame.transform.flip(pygame.transform.rotate(pygame.image.load(image_file), 180), 1, 0)
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = [offset_x, height - constants.PIPE_HEIGHT] if origin == 'top' else [offset_x, offset_y]
        self.mask = pygame.mask.from_surface(self.image)
        self.id = Pipe.count

    def scrollToleft(self):
        self.rect.left -= constants.SPEED
