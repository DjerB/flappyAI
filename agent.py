import pygame
import time
import constants


class Agent:
    def __init__(self, image_file):
        pygame.sprite.Sprite.__init__(self)  # call Sprite initializer
        self.vertical_speed = 0
        self.flapping = False
        self.timeStep = time.time()
        self.image = pygame.transform.scale(pygame.image.load(image_file), (constants.AGENT_WIDTH, constants.AGENT_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = [int(constants.WINDOW_WIDTH/10), int(constants.WINDOW_HEIGHT/2)]
        self.mask = pygame.mask.from_surface(self.image)
        self.size = self.image.get_rect().size

    def reset(self):
        self.timeStep = time.time()
        self.vertical_speed = 0
        self.rect.left, self.rect.top = [int(constants.WINDOW_WIDTH / 10), int(constants.WINDOW_HEIGHT / 2)]

    def resume(self, fps):
        self.timeStep = time.time() - (1 / fps)

    def flap(self):
        self.flapping = True

    def bounce(self):
        self.vertical_speed = 0

    def move(self):
        out_of_bounds = False
        duration = (time.time() - self.timeStep) * 10
        if self.flapping:
            self.vertical_speed = - constants.FLAPPING
            self.flapping = False
        if self.rect.top + self.size[1] + self.vertical_speed * duration < 0 or self.rect.top + self.vertical_speed * duration > constants.WINDOW_HEIGHT:
            out_of_bounds = True
        #self.rect.top = max(0, min(constants.WINDOW_HEIGHT - self.size[1], self.rect.top + self.vertical_speed * duration))
        self.rect.top = self.rect.top + self.vertical_speed * duration
        self.vertical_speed += constants.GRAVITY * duration
        self.timeStep = time.time()
        return out_of_bounds
