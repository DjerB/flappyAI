import pygame
import constants


class Score(pygame.sprite.Sprite):
    def __init__(self, location):
        pygame.sprite.Sprite.__init__(self)  #call Sprite initializer
        self.init_location = location
        self.score_digits = ['0']
        self.digits = [pygame.image.load('assets/0.png')]
        self.surface = pygame.Surface((constants.DIGIT_WIDTH, constants.DIGIT_HEIGHT))
        self.rect = self.surface.get_rect()
        self.rect.left, self.rect.top = location
        self.draw_digits()

    def draw_digits(self):
        width = 0
        for digit in self.score_digits:
            width += constants.ONE_WIDTH if digit == '1' else constants.DIGIT_WIDTH
        self.surface = pygame.Surface((width, constants.DIGIT_HEIGHT))
        self.rect.left = self.init_location[0] - (constants.DIGIT_WIDTH + 2) * (len(self.digits) - 1)
        offset = 0
        for idx, digit in enumerate(self.digits):
            rect = digit.get_rect()
            rect.left, rect.top = offset, 0
            self.surface.blit(digit, rect)
            offset += constants.ONE_WIDTH if self.score_digits[idx] == '1' else constants.DIGIT_WIDTH

    def update(self, score):
        self.score_digits = list(str(score))
        self.digits = []
        for digit in self.score_digits:
            self.digits.append(pygame.image.load(f'assets/{digit}.png'))
        self.draw_digits()
