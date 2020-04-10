import pygame
from agent import Agent
from pipe import Pipe
from background import Background
from welcome import Welcome
from gameover import GameOver
from pause import Pause
from play import Play
from score import Score
import utils
import constants
import random
import numpy as np


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption('Flappy Bird AI')
        self.pipes = []
        self.passed_pipes = set()
        # Set the HEIGHT and WIDTH of the screen
        self.screen = pygame.display.set_mode([constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT])
        # Set background
        self.background_1 = Background('assets/bg.png', [0, 0])
        self.background_2 = Background('assets/bg.png', [constants.WINDOW_WIDTH, 0])
        self.background_3 = Background('assets/bg.png', [2*constants.WINDOW_WIDTH, 0])
        self.base_1 = Background('assets/base.png', [0, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.base_2 = Background('assets/base.png', [constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.base_3 = Background('assets/base.png', [2 * constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.screen.blit(self.background_1.image, self.background_1.rect)
        self.screen.blit(self.background_2.image, self.background_2.rect)
        self.screen.blit(self.background_3.image, self.background_3.rect)
        self.screen.blit(self.base_1.image, self.base_1.rect)
        self.screen.blit(self.base_2.image, self.base_2.rect)
        self.screen.blit(self.base_3.image, self.base_3.rect)
        # Set agent
        self.agent = Agent('assets/bird.png')
        self.meter_counts = [0, 0]
        self.screen.blit(self.agent.image, self.agent.rect)
        # Set speed
        self.fps = 60
        # Used to manage how fast the screen updates
        self.clock = pygame.time.Clock()
        # Keep track of number of games
        self.nb_games = 0
        self.score = 0

        # For AI agent
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        self.state = np.moveaxis(utils.grayscale(state), -1, 0)
        self.reward = 0

        self.init_pipes()

    def reset(self):
        # Set background
        self.background_1 = Background('assets/bg.png', [0, 0])
        self.background_2 = Background('assets/bg.png', [constants.WINDOW_WIDTH, 0])
        self.background_3 = Background('assets/bg.png', [2 * constants.WINDOW_WIDTH, 0])
        self.base_1 = Background('assets/base.png', [0, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.base_2 = Background('assets/base.png',
                                 [constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.base_3 = Background('assets/base.png',
                                 [2 * constants.WINDOW_WIDTH, constants.WINDOW_HEIGHT - constants.BASE_HEIGHT])
        self.screen.blit(self.background_1.image, self.background_1.rect)
        self.screen.blit(self.background_2.image, self.background_2.rect)
        self.screen.blit(self.background_3.image, self.background_3.rect)
        self.screen.blit(self.base_1.image, self.base_1.rect)
        self.screen.blit(self.base_2.image, self.base_2.rect)
        self.screen.blit(self.base_3.image, self.base_3.rect)
        self.agent.reset()
        self.screen.blit(self.agent.image, self.agent.rect)
        self.meter_counts = [0, 0]
        self.score = 0
        self.init_pipes()

    def generate_random_pipes_pair(self):
        min_offset = max([pipes_pair[0].rect.left for pipes_pair in self.pipes]) if len(self.pipes) > 0 \
            else self.agent.rect.left + self.agent.size[0] + int(constants.WINDOW_WIDTH / 10)
        min_offset += (2 * constants.PIPE_WIDTH)
        offset = random.randint(min_offset, min_offset + constants.WINDOW_WIDTH)
        gap = random.randint(self.agent.size[1] * 2 + constants.FLAPPING, int(constants.WINDOW_HEIGHT / 2))
        height = constants.PIPE_HEIGHT + 1
        while height > constants.PIPE_HEIGHT or (constants.WINDOW_HEIGHT - height - gap) > constants.PIPE_HEIGHT:
            height = random.randint(constants.PIPE_MIN_HEIGHT,
                                    constants.WINDOW_HEIGHT - constants.PIPE_MIN_HEIGHT - gap)
        return Pipe('assets/pipe.png', height=height, offset_x=offset), \
               Pipe('assets/pipe.png', height=(constants.WINDOW_HEIGHT - gap - height), offset_x=offset, offset_y=(gap + height), origin='bottom')

    def init_pipes(self):
        self.pipes = []
        self.passed_pipes = set()
        nb_pipes = random.randint(constants.MIN_NB_PIPES, constants.MAX_NB_PIPES)
        for i in range(nb_pipes):
            self.pipes.append(self.generate_random_pipes_pair())

    def update_pipes(self):
        if len(self.pipes) > 0:
            if self.pipes[0][0].rect.left + self.pipes[0][0].rect.width < 0:
                self.pipes.pop(0)
        add_pipe = bool(random.getrandbits(1))
        if add_pipe:
            self.pipes.append(self.generate_random_pipes_pair())

    def redraw_screen(self, move=True):
        if move:
            self.background_1.scrollToleft()
            self.background_2.scrollToleft()
            self.background_3.scrollToleft()
            self.base_1.scrollToleft()
            self.base_2.scrollToleft()
            self.base_3.scrollToleft()
        self.screen.blit(self.background_1.image, self.background_1.rect)
        self.screen.blit(self.background_2.image, self.background_2.rect)
        self.screen.blit(self.background_3.image, self.background_3.rect)
        self.screen.blit(self.base_1.image, self.base_1.rect)
        self.screen.blit(self.base_2.image, self.base_2.rect)
        self.screen.blit(self.base_3.image, self.base_3.rect)
        for pipes_pair in self.pipes:
            if move:
                pipes_pair[0].scrollToleft()
                pipes_pair[1].scrollToleft()
            self.screen.blit(pipes_pair[0].image, pipes_pair[0].rect)
            self.screen.blit(pipes_pair[1].image, pipes_pair[1].rect)
        self.screen.blit(self.agent.image, self.agent.rect)

    def check_collision(self):
        between_pipes = False
        is_collision = False
        for pipes_pair in self.pipes:
            top_lim, bottom_lim = pipes_pair[0].rect.top + pipes_pair[0].rect.height, pipes_pair[1].rect.top - \
                                  self.agent.size[1]
            # If the pipe is already behind, ignore it
            if self.agent.rect.left + self.agent.size[0] > pipes_pair[0].rect.left and top_lim <= self.agent.rect.top <= bottom_lim:
                self.passed_pipes.add(pipes_pair[0].id)
                self.score = len(self.passed_pipes)
                if self.agent.rect.left < pipes_pair[0].rect.left + constants.PIPE_WIDTH:
                    between_pipes = True
                    if abs(self.agent.rect.top - top_lim) < 2 or abs(self.agent.rect.top - bottom_lim) < 2:
                        self.agent.bounce()
                    break
                continue
            else:
                top_pipe_mask = pipes_pair[0].mask
                bottom_pipe_mask = pipes_pair[1].mask
                agent_mask = self.agent.mask
                top_offset_x, top_offset_y = (self.agent.rect.left - pipes_pair[0].rect.left), (
                            self.agent.rect.top - pipes_pair[0].rect.top)
                bottom_offset_x, bottom_offset_y = (self.agent.rect.left - pipes_pair[1].rect.left), (
                        self.agent.rect.top - pipes_pair[1].rect.top)
                if top_pipe_mask.overlap(agent_mask, (top_offset_x, top_offset_y)) or bottom_pipe_mask.overlap(agent_mask, (bottom_offset_x, bottom_offset_y)):
                    is_collision = True
                    break
        return is_collision, between_pipes

    # FOR AI AGENT
    def frame_step(self, action=0):
        for event in pygame.event.get():  # User did something
            if event.type == pygame.QUIT:  # If user clicked close
                done = True  # Flag that we are done so we exit this loop
        # Reward for any frame
        reward = 0.1

        if action == 1:
            self.agent.flap()

        out_of_bounds = self.agent.move()

        # Update pipes as the agent moves forward
        self.meter_counts[1] += constants.SPEED
        if self.meter_counts[1] - self.meter_counts[0] > constants.PIPE_WIDTH:
            self.meter_counts[0] = self.meter_counts[1]
            self.update_pipes()

        # Check if the agent crashed
        collided, between_pipes = self.check_collision()

        if between_pipes:
            reward = 1

        # Terminal state if bird is out of the screen or crashed
        collided = collided or out_of_bounds

        if collided:
            reward = - 1
            self.reset()

        self.redraw_screen()

        # Limit the number of frames per second
        self.clock.tick(self.fps)

        # Get the array of pixels
        state = pygame.surfarray.array3d(pygame.display.get_surface())
        state = np.moveaxis(utils.grayscale(state), -1, 0)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        self.state = state
        self.reward = reward

        #return state, reward,

    def play(self):
        state = constants.STATE_START
        score_board = Score(location=(constants.WINDOW_WIDTH - 30, 20))
        play_button = Play(location=(int((constants.WINDOW_WIDTH - constants.PLAY_WIDTH) / 2),
                                          int((constants.WINDOW_HEIGHT - constants.PLAY_HEIGHT) / 2)))
        pause_button = Pause(location=(20, 20))
        welcome_message = Welcome(location=(int((constants.WINDOW_WIDTH - constants.MESSAGE_WIDTH) / 2),
                                            int((constants.WINDOW_HEIGHT - constants.MESSAGE_HEIGHT) / 2)))
        gameover_message = GameOver(location=(int((constants.WINDOW_WIDTH - constants.GAMEOVER_WIDTH) / 2),
                                              int((constants.WINDOW_HEIGHT - constants.GAMEOVER_HEIGHT) / 3)))
        collided = False
        done = False
        while not done:
            for event in pygame.event.get():  # User did something
                if event.type == pygame.QUIT:  # If user clicked close
                    done = True  # Flag that we are done so we exit this loop
                elif event.type == pygame.KEYDOWN:
                    if state == constants.STATE_START:
                        if event.key == pygame.K_SPACE:
                            state = constants.STATE_RUN
                            self.nb_games += 1
                            if self.nb_games > 1:
                                self.reset()
                    elif state == constants.STATE_RUN:
                        if event.key == pygame.K_SPACE:
                            self.agent.flap()
                        elif event.key == pygame.K_ESCAPE:
                            state = constants.STATE_PAUSE
                    elif state == constants.STATE_PAUSE:
                        if event.key == pygame.K_SPACE:
                            if collided:
                                state = constants.STATE_RUN
                                self.nb_games += 1
                                if self.nb_games > 1:
                                    self.reset()
                            else:
                                self.agent.resume(fps=self.fps)
                                state = constants.STATE_RUN
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = event.pos  # gets mouse position

                    if state == constants.STATE_RUN:
                        if pause_button.rect.collidepoint(mouse_pos):
                            state = constants.STATE_PAUSE
                        else:
                            self.agent.flap()
                    elif state == constants.STATE_START and welcome_message.rect.collidepoint(mouse_pos):
                        state = constants.STATE_RUN
                        self.nb_games += 1
                        if self.nb_games > 1:
                            self.reset()

                    elif state == constants.STATE_PAUSE:
                        if gameover_message.rect.collidepoint(mouse_pos):
                            state = constants.STATE_START
                            collided = False
                        elif play_button.rect.collidepoint(mouse_pos):
                            self.agent.resume(fps=self.fps)
                            state = constants.STATE_RUN

            if state == constants.STATE_START:
                self.redraw_screen(move=False)
                self.screen.blit(welcome_message.image, welcome_message.rect)

            elif state == constants.STATE_RUN:
                out_of_bounds = self.agent.move()
                self.meter_counts[1] += constants.SPEED

                if self.meter_counts[1] - self.meter_counts[0] > constants.PIPE_WIDTH:
                    self.meter_counts[0] = self.meter_counts[1]
                    self.update_pipes()

                collided, _ = self.check_collision()

                collided = collided or out_of_bounds

                if collided:
                    state = constants.STATE_PAUSE

                score_board.update(score=self.score)

                self.redraw_screen()
                self.screen.blit(pause_button.image, pause_button.rect)
                self.screen.blit(score_board.surface, score_board.rect)

            elif state == constants.STATE_PAUSE:
                self.redraw_screen(move=False)
                if collided:
                    self.screen.blit(gameover_message.image, gameover_message.rect)
                else:
                    self.screen.blit(play_button.image, play_button.rect)
            # Limit the number of frames per second
            fps = self.fps if state == constants.STATE_RUN else int(self.fps / 2)
            self.clock.tick(fps)

            # Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

        # Be IDLE friendly. If you forget this line, the program will 'hang'
        # on exit.
        pygame.quit()