from itertools import cycle
import random
import sys
import os
import pygame
from pygame.locals import *
import numpy as np

from Genetics.populationclass import Population
from Genetics.birdnetclass import BirdNet

# Display settings
SCREENWIDTH = 288
SCREENHEIGHT = 512
PIPEGAPSIZE = 130  # Vertical gap between upper and lower pipe
PIPE_SPACING = 200  # Horizontal distance between pipe pairs
BASEY = SCREENHEIGHT * 0.79

# Training settings
RENDER_EVERY = 5

# Physics - tuned for proper flappy feel
GRAVITY = 0.5
FLAP_STRENGTH = -8
MAX_FALL_SPEED = 8
PIPE_SPEED = -3

# Game states
STATE_MENU = 'menu'
STATE_MODEL_SELECT = 'model_select'
STATE_TRAIN_SETTINGS = 'train_settings'
STATE_PLAY = 'play'
STATE_RUN_MODEL = 'run_model'
STATE_TRAIN = 'train'

# Default training settings
DEFAULT_SETTINGS = {
    'population': 50,
    'mutation_rate': 0.1,
    'parent_fraction': 0.3,
    'hidden_structure': [4],  # List of hidden layer sizes
    'runs_per_bird': 3,  # Evaluate fitness over N runs
    'fitness_method': 'min',  # How to combine runs: 'avg', 'min', 'geo', 'harm'
}

FITNESS_METHODS = ['avg', 'min', 'geo', 'harm']
FITNESS_LABELS = {
    'avg': 'Average',
    'min': 'Minimum',
    'geo': 'Geometric',
    'harm': 'Harmonic',
}

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), 'training_settings.json')


def load_training_settings():
    """Load persisted training settings or return defaults."""
    try:
        import json
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_SETTINGS.copy()


def save_training_settings(settings):
    """Persist training settings to file."""
    import json
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)

# Assets
IMAGES, SOUNDS, HITMASKS = {}, {}, {}
address_assets = 'assets/sprites/'
models_dir = 'models/'

# Ensure models directory exists
os.makedirs(models_dir, exist_ok=True)

PLAYERS_LIST = (
    (address_assets + 'redbird-upflap.png', address_assets + 'redbird-midflap.png', address_assets + 'redbird-downflap.png'),
    (address_assets + 'bluebird-upflap.png', address_assets + 'bluebird-midflap.png', address_assets + 'bluebird-downflap.png'),
    (address_assets + 'yellowbird-upflap.png', address_assets + 'yellowbird-midflap.png', address_assets + 'yellowbird-downflap.png'),
)
BACKGROUNDS_LIST = (address_assets + 'background-day.png', address_assets + 'background-night.png')
PIPES_LIST = (address_assets + 'pipe-green.png', address_assets + 'pipe-red.png')


# Cache for scaled surface
_scale_cache = {'size': None, 'surface': None}

def scale_to_screen(surface, screen):
    """Scale game surface to fit current window size, with caching."""
    current_size = screen.get_size()

    # Only create new scaled surface if window size changed
    if _scale_cache['size'] != current_size:
        _scale_cache['size'] = current_size
        _scale_cache['surface'] = pygame.Surface(current_size)

    # Scale directly into cached surface
    pygame.transform.scale(surface, current_size, _scale_cache['surface'])
    return _scale_cache['surface']


def get_saved_models():
    """Return list of saved model files."""
    if not os.path.exists(models_dir):
        return []
    return [f[:-4] for f in os.listdir(models_dir) if f.endswith('.npz')]


def save_model(bird, name, settings=None):
    """Save a bird's weights to file, optionally with training settings."""
    import json
    filepath = os.path.join(models_dir, f"{name}.npz")
    np.savez(filepath, *bird.tensors)
    print(f"Model saved to {filepath}")

    # Save companion settings file if provided
    if settings:
        settings_path = os.path.join(models_dir, f"{name}.json")
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)


def load_model_settings(name):
    """Load training settings for a model, if available."""
    import json
    settings_path = os.path.join(models_dir, f"{name}.json")
    try:
        with open(settings_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_model(name):
    """Load weights into a new BirdNet, inferring structure from saved tensors."""
    filepath = os.path.join(models_dir, f"{name}.npz")
    data = np.load(filepath)
    tensors = [data[f'arr_{i}'] for i in range(len(data.files))]

    # Infer network structure from tensor shapes
    # tensor[i] has shape (next_layer_size, current_layer_size)
    structure = [tensors[0].shape[1]]  # input size
    for t in tensors:
        structure.append(t.shape[0])  # each layer's output size

    # Create bird with inferred structure
    bird = BirdNet()
    bird.tensors = tensors
    # Rebuild vectors to match loaded structure
    bird.vectors = [np.zeros(shape=(n, 1)) for n in structure]
    return bird


def load_assets():
    """Load all game assets."""
    # Number sprites
    IMAGES['numbers'] = tuple(
        pygame.image.load(address_assets + f'{i}.png').convert_alpha()
        for i in range(10)
    )
    IMAGES['message'] = pygame.image.load(address_assets + 'message.png').convert_alpha()
    IMAGES['base'] = pygame.image.load(address_assets + 'base.png').convert_alpha()
    IMAGES['dot'] = pygame.image.load(address_assets + 'dot.png').convert_alpha()
    IMAGES['bird_dot'] = pygame.image.load(address_assets + 'bird_dot.png').convert_alpha()
    IMAGES['gameover'] = pygame.image.load(address_assets + 'gameover.png').convert_alpha()


def load_random_sprites():
    """Load random background, pipe, and player sprites."""
    IMAGES['background'] = pygame.image.load(random.choice(BACKGROUNDS_LIST)).convert()
    pipe_img = pygame.image.load(random.choice(PIPES_LIST)).convert_alpha()
    IMAGES['pipe'] = (pygame.transform.rotate(pipe_img, 180), pipe_img)
    player_set = random.choice(PLAYERS_LIST)
    IMAGES['player'] = tuple(pygame.image.load(p).convert_alpha() for p in player_set)

    HITMASKS['pipe'] = (getHitmask(IMAGES['pipe'][0]), getHitmask(IMAGES['pipe'][1]))
    HITMASKS['player'] = tuple(getHitmask(img) for img in IMAGES['player'])


def draw_text(surface, text, x, y, font, color=(255, 255, 255), center=False):
    """Draw text on surface."""
    rendered = font.render(text, True, color)
    if center:
        x = x - rendered.get_width() // 2
    surface.blit(rendered, (x, y))


def main_menu(screen, game_surface, font, font_large):
    """Main menu screen."""
    options = ['PLAY GAME', 'RUN MODEL', 'TRAIN', 'EXIT']
    selected = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == K_RETURN:
                    if options[selected] == 'PLAY GAME':
                        return STATE_PLAY, None
                    elif options[selected] == 'RUN MODEL':
                        return STATE_MODEL_SELECT, 'run'
                    elif options[selected] == 'TRAIN':
                        return STATE_MODEL_SELECT, 'train'
                    elif options[selected] == 'EXIT':
                        pygame.quit()
                        sys.exit()
                elif event.key == K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        # Draw menu
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        # Title
        draw_text(game_surface, 'FLAPPY BIRD', SCREENWIDTH // 2, 80, font_large, (255, 255, 255), center=True)
        draw_text(game_surface, 'ML', SCREENWIDTH // 2, 120, font_large, (255, 200, 0), center=True)

        # Menu options
        for i, option in enumerate(options):
            color = (255, 255, 0) if i == selected else (255, 255, 255)
            prefix = '> ' if i == selected else '  '
            draw_text(game_surface, prefix + option, SCREENWIDTH // 2, 200 + i * 40, font, color, center=True)

        # Scale and display
        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def delete_model(name):
    """Delete a saved model and its settings."""
    filepath = os.path.join(models_dir, f"{name}.npz")
    settings_path = os.path.join(models_dir, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
    if os.path.exists(settings_path):
        os.remove(settings_path)


def model_select_menu(screen, game_surface, font, font_large, mode):
    """Model selection submenu."""
    models = get_saved_models()
    options = ['NEW (random weights)'] + models + ['BACK']
    selected = 0
    clock = pygame.time.Clock()

    title = 'SELECT MODEL TO RUN' if mode == 'run' else 'SELECT STARTING WEIGHTS'

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == K_RETURN:
                    if options[selected] == 'BACK':
                        return STATE_MENU, None
                    elif options[selected] == 'NEW (random weights)':
                        if mode == 'run':
                            return STATE_RUN_MODEL, None
                        else:
                            return STATE_TRAIN_SETTINGS, None
                    else:
                        if mode == 'run':
                            return STATE_RUN_MODEL, options[selected]
                        else:
                            return STATE_TRAIN_SETTINGS, options[selected]
                elif event.key == K_ESCAPE:
                    return STATE_MENU, None
                elif event.key == K_d or event.key == K_DELETE or event.key == K_BACKSPACE:
                    # Delete selected model (not NEW or BACK)
                    if options[selected] not in ['NEW (random weights)', 'BACK']:
                        delete_model(options[selected])
                        # Refresh list
                        models = get_saved_models()
                        options = ['NEW (random weights)'] + models + ['BACK']
                        selected = min(selected, len(options) - 1)

        # Draw
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        draw_text(game_surface, title, SCREENWIDTH // 2, 60, font, (255, 255, 255), center=True)

        # Scrollable list (show max 8 items)
        start_idx = max(0, selected - 4)
        end_idx = min(len(options), start_idx + 8)

        for i, idx in enumerate(range(start_idx, end_idx)):
            option = options[idx]
            color = (255, 255, 0) if idx == selected else (255, 255, 255)
            prefix = '> ' if idx == selected else '  '
            draw_text(game_surface, prefix + option[:20], SCREENWIDTH // 2, 120 + i * 35, font, color, center=True)

        # Show settings for selected model
        if options[selected] not in ['NEW (random weights)', 'BACK']:
            model_settings = load_model_settings(options[selected])
            if model_settings:
                struct = model_settings.get('hidden_structure', [4])
                struct_str = '-'.join(str(s) for s in struct)
                fitness = FITNESS_LABELS.get(model_settings.get('fitness_method', '?'), '?')
                info1 = f"Net: 3-{struct_str}-1 | Pop: {model_settings.get('population', '?')}"
                info2 = f"Mut: {model_settings.get('mutation_rate', '?')} | Fitness: {fitness}"
                draw_text(game_surface, info1, SCREENWIDTH // 2, SCREENHEIGHT - 95, font, (180, 180, 180), center=True)
                draw_text(game_surface, info2, SCREENWIDTH // 2, SCREENHEIGHT - 75, font, (180, 180, 180), center=True)
            else:
                draw_text(game_surface, '(no settings saved)', SCREENWIDTH // 2, SCREENHEIGHT - 85, font, (120, 120, 120), center=True)
            draw_text(game_surface, '[D] Delete', SCREENWIDTH // 2, SCREENHEIGHT - 50, font, (200, 100, 100), center=True)

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def name_model_screen(screen, game_surface, font, font_large, default_name=""):
    """Screen to input a name for the model."""
    name = default_name
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_RETURN:
                    return name if name else default_name
                elif event.key == K_ESCAPE:
                    return default_name  # Use default if cancelled
                elif event.key == K_BACKSPACE:
                    name = name[:-1]
                elif event.unicode.isalnum() or event.unicode in '-_':
                    if len(name) < 30:
                        name += event.unicode

        # Draw
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        draw_text(game_surface, 'SAVE MODEL', SCREENWIDTH // 2, 80, font_large, (255, 255, 255), center=True)
        draw_text(game_surface, 'Enter name:', SCREENWIDTH // 2, 150, font, (200, 200, 200), center=True)

        # Input box
        display_name = name + '_' if len(name) < 30 else name
        draw_text(game_surface, display_name, SCREENWIDTH // 2, 190, font, (255, 255, 0), center=True)

        draw_text(game_surface, 'ENTER to save | ESC for default', SCREENWIDTH // 2, 250, font, (150, 150, 150), center=True)

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def train_settings_menu(screen, game_surface, font, font_large, model_name=None):
    """Training parameters menu with text input support."""
    # Load persisted settings
    saved = load_training_settings()
    settings = {
        'population': str(saved.get('population', 50)),
        'mutation_rate': str(saved.get('mutation_rate', 0.1)),
        'parent_fraction': saved.get('parent_fraction', 0.3),
        'hidden_structure': saved.get('hidden_structure', [4]),
        'runs_per_bird': str(saved.get('runs_per_bird', 3)),
        'fitness_method': saved.get('fitness_method', 'min'),
    }

    # If loading a model, infer structure from it and lock it
    structure_locked = False
    if model_name:
        try:
            loaded_model = load_model(model_name)
            # Infer hidden structure from tensors (exclude input and output layers)
            inferred_structure = [t.shape[0] for t in loaded_model.tensors[:-1]]
            settings['hidden_structure'] = inferred_structure
            structure_locked = True
        except Exception:
            pass  # Fall back to editable structure if load fails

    # Menu items: (key, label, type)
    # type: 'int_slider', 'float_slider', 'text', 'text_int', 'structure', 'selector'
    items = [
        ('population', 'Population', 'text_int'),
        ('mutation_rate', 'Mutation Rate', 'text'),
        ('parent_fraction', 'Parent %', 'float_slider', 0.1, 0.9, 0.1),
        ('hidden_structure', 'Network', 'structure'),
        ('runs_per_bird', 'Runs/Bird', 'text_int'),
        ('fitness_method', 'Fitness', 'selector', FITNESS_METHODS),
    ]

    # Hints for each setting
    hints = {
        'population': 'More = diverse gene pool, slower',
        'mutation_rate': 'Higher = explore more, lower = refine',
        'parent_fraction': 'Higher = keep more, lower = selective',
        'hidden_structure': 'More nodes = smarter but harder to train',
        'runs_per_bird': 'More = reliable fitness, slower',
        'fitness_method': 'min=consistent, avg=overall, geo/harm=balanced',
    }

    selected = 0
    editing_text = False
    editing_structure = False
    structure_cursor = 0  # Which layer we're editing
    clock = pygame.time.Clock()

    total_items = len(items) + 1  # +1 for START

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

            if event.type == KEYDOWN:
                if editing_text:
                    # Text input mode
                    item = items[selected]
                    key = item[0]
                    if event.key == K_RETURN or event.key == K_ESCAPE:
                        editing_text = False
                    elif event.key == K_BACKSPACE:
                        settings[key] = settings[key][:-1]
                    elif item[2] == 'text_int' and event.unicode in '0123456789':
                        settings[key] += event.unicode
                    elif item[2] == 'text' and event.unicode in '0123456789.':
                        settings[key] += event.unicode

                elif editing_structure:
                    # Structure editing mode
                    struct = settings['hidden_structure']
                    if event.key == K_ESCAPE or event.key == K_RETURN:
                        editing_structure = False
                    elif event.key == K_LEFT:
                        structure_cursor = max(0, structure_cursor - 1)
                    elif event.key == K_RIGHT:
                        structure_cursor = min(len(struct), structure_cursor + 1)
                    elif event.key == K_UP:
                        if structure_cursor < len(struct):
                            struct[structure_cursor] = min(64, struct[structure_cursor] + 1)
                    elif event.key == K_DOWN:
                        if structure_cursor < len(struct):
                            struct[structure_cursor] = max(1, struct[structure_cursor] - 1)
                    elif event.key == K_a:  # Add layer
                        if len(struct) < 5:
                            struct.insert(structure_cursor, 4)
                    elif event.key == K_d or event.key == K_DELETE:  # Delete layer
                        if len(struct) > 1 and structure_cursor < len(struct):
                            struct.pop(structure_cursor)
                            structure_cursor = min(structure_cursor, len(struct) - 1)

                else:
                    # Normal navigation
                    if event.key == K_UP:
                        selected = (selected - 1) % total_items
                    elif event.key == K_DOWN:
                        selected = (selected + 1) % total_items
                    elif event.key == K_RETURN:
                        if selected == len(items):  # START
                            # Convert text inputs to proper types
                            try:
                                settings['mutation_rate'] = float(settings['mutation_rate'])
                            except ValueError:
                                settings['mutation_rate'] = 0.1
                            try:
                                settings['population'] = max(10, int(settings['population']))
                            except ValueError:
                                settings['population'] = 50
                            try:
                                settings['runs_per_bird'] = max(1, int(settings['runs_per_bird']))
                            except ValueError:
                                settings['runs_per_bird'] = 3
                            # Save settings for next time
                            save_training_settings(settings)
                            return STATE_TRAIN, settings
                        elif items[selected][2] in ('text', 'text_int'):
                            editing_text = True
                        elif items[selected][2] == 'structure' and not structure_locked:
                            editing_structure = True
                            structure_cursor = 0
                    elif event.key == K_LEFT and selected < len(items):
                        item = items[selected]
                        if item[2] == 'int_slider':
                            settings[item[0]] = max(item[3], settings[item[0]] - item[5])
                        elif item[2] == 'float_slider':
                            settings[item[0]] = max(item[3], round(settings[item[0]] - item[5], 2))
                        elif item[2] == 'selector':
                            options = item[3]
                            idx = options.index(settings[item[0]])
                            settings[item[0]] = options[(idx - 1) % len(options)]
                    elif event.key == K_RIGHT and selected < len(items):
                        item = items[selected]
                        if item[2] == 'int_slider':
                            settings[item[0]] = min(item[4], settings[item[0]] + item[5])
                        elif item[2] == 'float_slider':
                            settings[item[0]] = min(item[4], round(settings[item[0]] + item[5], 2))
                        elif item[2] == 'selector':
                            options = item[3]
                            idx = options.index(settings[item[0]])
                            settings[item[0]] = options[(idx + 1) % len(options)]
                    elif event.key == K_ESCAPE:
                        return STATE_MENU, None

        # Draw
        game_surface.blit(IMAGES['background'], (0, 0))
        game_surface.blit(IMAGES['base'], (0, BASEY))

        draw_text(game_surface, 'TRAINING SETTINGS', SCREENWIDTH // 2, 40, font, (255, 255, 255), center=True)

        y_pos = 80
        for i, item in enumerate(items):
            key, label, item_type = item[0], item[1], item[2]
            is_selected = (i == selected)
            color = (255, 255, 0) if is_selected else (255, 255, 255)
            prefix = '> ' if is_selected else '  '

            if item_type == 'structure':
                struct = settings['hidden_structure']
                struct_str = '3-' + '-'.join(str(s) for s in struct) + '-1'
                if structure_locked:
                    # Show locked structure (from loaded model)
                    locked_color = (120, 120, 120) if not is_selected else (180, 180, 100)
                    draw_text(game_surface, f'{prefix}{label}: {struct_str} (locked)', SCREENWIDTH // 2, y_pos, font, locked_color, center=True)
                elif editing_structure and is_selected:
                    # Show editable structure
                    struct_str = '3-['
                    for j, s in enumerate(struct):
                        if j == structure_cursor:
                            struct_str += f'[{s}]'
                        else:
                            struct_str += str(s)
                        if j < len(struct) - 1:
                            struct_str += '-'
                    struct_str += ']-1'
                    draw_text(game_surface, f'{prefix}{label}: {struct_str}', SCREENWIDTH // 2, y_pos, font, (100, 255, 100), center=True)
                    draw_text(game_surface, 'UP/DOWN:size A:add D:del', SCREENWIDTH // 2, y_pos + 18, font, (150, 150, 150), center=True)
                    y_pos += 20
                else:
                    draw_text(game_surface, f'{prefix}{label}: {struct_str}', SCREENWIDTH // 2, y_pos, font, color, center=True)

            elif item_type in ('text', 'text_int'):
                val = settings[key]
                if editing_text and is_selected:
                    draw_text(game_surface, f'{prefix}{label}: {val}_', SCREENWIDTH // 2, y_pos, font, (100, 255, 100), center=True)
                else:
                    draw_text(game_surface, f'{prefix}{label}: {val}', SCREENWIDTH // 2, y_pos, font, color, center=True)

            elif item_type == 'selector':
                val = settings[key]
                display_val = FITNESS_LABELS.get(val, val)
                draw_text(game_surface, f'{prefix}{label}: < {display_val} >', SCREENWIDTH // 2, y_pos, font, color, center=True)

            else:  # sliders
                val = settings[key]
                if isinstance(val, float):
                    val_str = f'{val:.2f}'
                else:
                    val_str = str(int(val))
                draw_text(game_surface, f'{prefix}{label}: {val_str}', SCREENWIDTH // 2, y_pos, font, color, center=True)

            y_pos += 35

        # Controls hint
        if not editing_text and not editing_structure:
            draw_text(game_surface, 'ENTER:edit  LEFT/RIGHT:adjust', SCREENWIDTH // 2, y_pos, font, (150, 150, 150), center=True)
        y_pos += 25

        # START button
        start_selected = (selected == len(items))
        start_color = (255, 255, 0) if start_selected else (255, 255, 255)
        start_prefix = '> ' if start_selected else '  '
        draw_text(game_surface, f'{start_prefix}START TRAINING', SCREENWIDTH // 2, y_pos + 10, font, start_color, center=True)

        # Setting-specific hint at bottom
        if selected < len(items):
            hint_key = items[selected][0]
            hint_text = hints.get(hint_key, '')
            draw_text(game_surface, hint_text, SCREENWIDTH // 2, SCREENHEIGHT - 35, font, (180, 180, 120), center=True)

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def play_game(screen, game_surface, font):
    """Manual play mode - single player controlled bird."""
    load_random_sprites()

    bird_x = SCREENWIDTH * 0.2
    bird_y = SCREENHEIGHT / 2
    bird_vel_y = 0

    basex = 0
    baseShift = IMAGES['base'].get_width() - SCREENWIDTH

    # Pipes
    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
    ]

    score = 0
    clock = pygame.time.Clock()

    while True:
        flap = False
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    return STATE_MENU
                if event.key == K_SPACE or event.key == K_UP:
                    flap = True

        # Flap
        if flap and bird_y > 0:
            bird_vel_y = FLAP_STRENGTH

        # Gravity
        if bird_vel_y < MAX_FALL_SPEED:
            bird_vel_y += GRAVITY
        bird_y += bird_vel_y

        # Move pipes
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += PIPE_SPEED
            lPipe['x'] += PIPE_SPEED

        # Add new pipe when last pipe has moved onto screen enough
        if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        # Remove off-screen pipe
        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)

        # Check score
        playerMidPos = bird_x + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1

        # Check crash
        crash = checkCrash({'x': bird_x, 'y': bird_y, 'index': 0}, upperPipes, lowerPipes)
        if crash[0]:
            return STATE_MENU

        # Draw
        basex = -((-basex + 4) % baseShift)
        game_surface.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        game_surface.blit(IMAGES['base'], (basex, BASEY))
        game_surface.blit(IMAGES['player'][1], (bird_x, bird_y))
        showScore(game_surface, score)

        draw_text(game_surface, 'ESC: Menu', 5, 5, font, (255, 255, 255))

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def run_model_game(screen, game_surface, font, model_name):
    """Run a trained model - watch single AI bird play."""
    load_random_sprites()

    # Load or create bird
    if model_name:
        bird = load_model(model_name)
    else:
        bird = BirdNet()

    bird.x = SCREENWIDTH * 0.2
    bird.y = SCREENHEIGHT / 2
    bird.birdVelY = 0

    basex = 0
    baseShift = IMAGES['base'].get_width() - SCREENWIDTH

    newPipe1 = getRandomPipe()
    newPipe2 = getRandomPipe()
    upperPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
    ]
    lowerPipes = [
        {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
    ]

    closest_pipe_x = upperPipes[0]['x']
    closest_pipe_y = lowerPipes[0]['y']

    score = 0
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                return STATE_MENU

        # AI decision
        bird.set_input((closest_pipe_x + 20) - (bird.x + 12), (bird.y + 12) - (closest_pipe_y - 50), bird.birdVelY)
        if bird.fly_up() and bird.y > 0:
            bird.birdVelY = FLAP_STRENGTH

        # Physics
        if bird.birdVelY < MAX_FALL_SPEED:
            bird.birdVelY += GRAVITY
        bird.y += bird.birdVelY

        # Move pipes
        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            uPipe['x'] += PIPE_SPEED
            lPipe['x'] += PIPE_SPEED
        closest_pipe_x += PIPE_SPEED

        if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
            newPipe = getRandomPipe()
            upperPipes.append(newPipe[0])
            lowerPipes.append(newPipe[1])

        if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upperPipes.pop(0)
            lowerPipes.pop(0)
            closest_pipe_x = upperPipes[0]['x']
            closest_pipe_y = lowerPipes[0]['y']

        # Score
        playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
        for pipe in upperPipes:
            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                score += 1
                if len(upperPipes) > 1:
                    closest_pipe_x = upperPipes[1]['x']
                    closest_pipe_y = lowerPipes[1]['y']

        # Crash
        crash = checkCrash({'x': bird.x, 'y': bird.y, 'index': 0}, upperPipes, lowerPipes)
        if crash[0]:
            return STATE_MENU

        # Draw
        basex = -((-basex + 4) % baseShift)
        game_surface.blit(IMAGES['background'], (0, 0))

        for uPipe, lPipe in zip(upperPipes, lowerPipes):
            game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
            game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

        game_surface.blit(IMAGES['base'], (basex, BASEY))
        game_surface.blit(IMAGES['player'][1], (bird.x, bird.y))
        showScore(game_surface, score)

        name_display = model_name if model_name else 'Random'
        draw_text(game_surface, f'Model: {name_display}', 5, 5, font, (255, 255, 255))
        draw_text(game_surface, 'ESC: Menu', 5, 25, font, (255, 255, 255))

        screen.blit(scale_to_screen(game_surface, screen), (0, 0))
        pygame.display.update()
        clock.tick(60)


def train_game(screen, game_surface, font, font_large, model_name, settings=None):
    """Training mode with evolution and multi-run evaluation."""
    import csv
    from datetime import datetime

    load_random_sprites()

    # Use provided settings or defaults
    if settings is None:
        settings = DEFAULT_SETTINGS.copy()

    runs_per_bird = settings.get('runs_per_bird', 3)

    # Initialize population with settings
    population = Population(
        bird_num=int(settings['population']),
        parent_fraction=settings['parent_fraction'],
        mutation_rate=settings['mutation_rate'],
        hidden_structure=settings['hidden_structure']
    )

    if model_name:
        # Load weights into all birds
        template = load_model(model_name)
        for idx, bird in enumerate(population.individuals):
            for i, tensor in enumerate(template.tensors):
                bird.tensors[i] = tensor.copy()
            if idx > 0:  # Keep first bird as exact copy, mutate the rest
                bird.mutate(population.mutation_rate)
        # Initialize best_ever with loaded model as baseline
        population.best_ever_weights = [t.copy() for t in template.tensors]

    birds = population.individuals
    generation = 1
    best_score_ever = 0
    best_score_distance = 0  # Distance when best score was achieved
    best_fitness_ever = 0
    best_fitness_score = 0   # Score when best fitness was achieved

    # Training log for graphs
    training_log = []

    clock = pygame.time.Clock()

    def save_and_generate_graphs():
        """Save model and generate training graphs."""
        # Save model
        if population.best_ever_weights is not None:
            best_bird = BirdNet()
            for i, tensor in enumerate(population.best_ever_weights):
                best_bird.tensors[i] = tensor.copy()
        else:
            population.sort()
            best_bird = population.individuals[0]

        # Prompt for model name
        default_name = f"gen{generation}_best{int(population.best_ever_distance)}"
        save_name = name_model_screen(screen, game_surface, font, font_large, default_name)
        save_model(best_bird, save_name, settings)

        # Create outputs directory structure: outputs/YYYY-MM-DD/
        now = datetime.now()
        outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs', now.strftime('%Y-%m-%d'))
        os.makedirs(outputs_dir, exist_ok=True)
        log_file = os.path.join(outputs_dir, f'{now.strftime("%H%M%S")}_{save_name}.csv')

        # Save CSV log
        if training_log:
            with open(log_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['generation', 'best_distance', 'mean_distance', 'sigma', 'best_score'])
                writer.writeheader()
                writer.writerows(training_log)
            print(f"Training log saved to {log_file}")

            # Generate graphs
            try:
                import matplotlib.pyplot as plt

                gens = [r['generation'] for r in training_log]
                best_dist = [r['best_distance'] for r in training_log]
                mean_dist = [r['mean_distance'] for r in training_log]
                sigma = [r['sigma'] for r in training_log]
                best_score = [r['best_score'] for r in training_log]

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Settings summary as subtitle
                struct_str = '-'.join(str(s) for s in settings['hidden_structure'])
                fitness_label = FITNESS_LABELS.get(settings.get('fitness_method', 'min'), 'min')
                settings_text = (f"Pop: {settings['population']} | Mut: {settings['mutation_rate']} | "
                                f"Parents: {settings['parent_fraction']*100:.0f}% | Net: 3-{struct_str}-1 | "
                                f"Runs: {runs_per_bird} | Fitness: {fitness_label}")
                fig.suptitle(settings_text, fontsize=9, color='gray', y=0.98)

                # Best distance over time
                axes[0, 0].plot(gens, best_dist, 'b-', linewidth=2)
                axes[0, 0].set_xlabel('Generation')
                axes[0, 0].set_ylabel('Distance')
                axes[0, 0].set_title('Best Distance per Generation')
                axes[0, 0].grid(True, alpha=0.3)

                # Mean distance with std band
                axes[0, 1].plot(gens, mean_dist, 'g-', linewidth=2, label='Mean')
                axes[0, 1].fill_between(gens,
                    [m - s for m, s in zip(mean_dist, sigma)],
                    [m + s for m, s in zip(mean_dist, sigma)],
                    alpha=0.3, color='green', label='±1 Sigma')
                axes[0, 1].set_xlabel('Generation')
                axes[0, 1].set_ylabel('Distance')
                axes[0, 1].set_title('Mean Distance ± Sigma')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

                # Best score (pipes passed)
                axes[1, 0].plot(gens, best_score, 'r-', linewidth=2)
                axes[1, 0].set_xlabel('Generation')
                axes[1, 0].set_ylabel('Score (Pipes)')
                axes[1, 0].set_title('Best Score per Generation')
                axes[1, 0].grid(True, alpha=0.3)

                # Sigma (diversity)
                axes[1, 1].plot(gens, sigma, 'm-', linewidth=2)
                axes[1, 1].set_xlabel('Generation')
                axes[1, 1].set_ylabel('Sigma')
                axes[1, 1].set_title('Population Diversity (Sigma)')
                axes[1, 1].grid(True, alpha=0.3)

                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave room for suptitle
                graph_file = log_file.replace('.csv', '.png')
                plt.savefig(graph_file, dpi=150)
                plt.close()
                print(f"Training graphs saved to {graph_file}")
            except ImportError:
                print("matplotlib not installed - skipping graph generation")

    while True:  # Generation loop
        # Multi-run evaluation: track distances per run for each bird
        all_run_distances = [[] for _ in birds]  # List of lists: bird -> [run1_dist, run2_dist, ...]
        cumulative_scores = [0 for _ in birds]
        gen_best_score = 0

        for run_idx in range(runs_per_bird):
            # Reset birds for this run
            for bird in birds:
                bird.x = SCREENWIDTH * 0.2
                bird.y = SCREENHEIGHT / 2
                bird.birdVelY = 0
                bird.score = 0
                bird.birdFlapped = False

            crashTest = [[False, False] for _ in birds]

            basex = 0
            baseShift = IMAGES['base'].get_width() - SCREENWIDTH

            newPipe1 = getRandomPipe()
            newPipe2 = getRandomPipe()
            upperPipes = [
                {'x': SCREENWIDTH + 200, 'y': newPipe1[0]['y']},
                {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[0]['y']},
            ]
            lowerPipes = [
                {'x': SCREENWIDTH + 200, 'y': newPipe1[1]['y']},
                {'x': SCREENWIDTH + 200 + PIPE_SPACING, 'y': newPipe2[1]['y']},
            ]

            closest_pipe_x = upperPipes[0]['x']
            closest_pipe_y = lowerPipes[0]['y']
            score = 0
            frame_count = 0
            run_distances = [0.0 for _ in birds]

            # Game loop for this run
            while True:
                frame_count += 1
                save_and_exit = False

                for event in pygame.event.get():
                    if event.type == QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == KEYDOWN and event.key == K_ESCAPE:
                        save_and_exit = True

                if save_and_exit:
                    save_and_generate_graphs()
                    return STATE_MENU

                # Bird AI decisions
                for bird in birds:
                    bird.set_input((closest_pipe_x + 20) - (bird.x + 12), (bird.y + 12) - (closest_pipe_y - 50), bird.birdVelY)
                    if bird.fly_up() and bird.y > 0:
                        bird.birdVelY = FLAP_STRENGTH
                        bird.birdFlapped = True

                # Check crashes
                for i, bird in enumerate(birds):
                    if not crashTest[i][0]:
                        crashTest[i] = checkCrash({'x': bird.x, 'y': bird.y, 'index': 0}, upperPipes, lowerPipes)

                # Check if all dead
                all_dead = all(ct[0] for ct in crashTest)
                if all_dead:
                    gen_best_score = max(gen_best_score, score)
                    break

                # Update birds
                for i, bird in enumerate(birds):
                    if not crashTest[i][0]:
                        run_distances[i] += abs(PIPE_SPEED)
                        if bird.birdVelY < MAX_FALL_SPEED and not bird.birdFlapped:
                            bird.birdVelY += GRAVITY
                        if bird.birdFlapped:
                            bird.birdFlapped = False
                        bird.y += bird.birdVelY

                # Move pipes
                for uPipe, lPipe in zip(upperPipes, lowerPipes):
                    uPipe['x'] += PIPE_SPEED
                    lPipe['x'] += PIPE_SPEED
                closest_pipe_x += PIPE_SPEED

                if upperPipes[-1]['x'] < SCREENWIDTH - PIPE_SPACING:
                    newPipe = getRandomPipe()
                    upperPipes.append(newPipe[0])
                    lowerPipes.append(newPipe[1])

                if upperPipes[0]['x'] < -IMAGES['pipe'][0].get_width():
                    upperPipes.pop(0)
                    lowerPipes.pop(0)
                    closest_pipe_x = upperPipes[0]['x']
                    closest_pipe_y = lowerPipes[0]['y']

                # Score check
                for i, bird in enumerate(birds):
                    if not crashTest[i][0]:
                        playerMidPos = bird.x + IMAGES['player'][0].get_width() / 2
                        for j, pipe in enumerate(upperPipes):
                            pipeMidPos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
                            if pipeMidPos <= playerMidPos < pipeMidPos + 4:
                                bird.score += 1
                                score = max(score, bird.score)
                                if len(upperPipes) > 1:
                                    closest_pipe_x = upperPipes[1]['x']
                                    closest_pipe_y = lowerPipes[1]['y']

                # Render every N frames
                if frame_count % RENDER_EVERY == 0:
                    basex = -((-basex + 20) % baseShift)
                    game_surface.blit(IMAGES['background'], (0, 0))

                    for uPipe, lPipe in zip(upperPipes, lowerPipes):
                        game_surface.blit(IMAGES['pipe'][0], (uPipe['x'], uPipe['y']))
                        game_surface.blit(IMAGES['pipe'][1], (lPipe['x'], lPipe['y']))

                    game_surface.blit(IMAGES['base'], (basex, BASEY))

                    # Draw alive birds
                    for i, bird in enumerate(birds):
                        if not crashTest[i][0]:
                            game_surface.blit(IMAGES['player'][1], (bird.x, bird.y))

                    showScore(game_surface, score)

                    # Stats
                    alive = sum(1 for ct in crashTest if not ct[0])
                    fitness_label = FITNESS_LABELS.get(settings.get('fitness_method', 'min'), 'Min')
                    draw_text(game_surface, f'Gen: {generation}  Run: {run_idx + 1}/{runs_per_bird}', 5, 5, font, (255, 255, 255))
                    draw_text(game_surface, f'Alive: {alive}/{len(birds)}', 5, 25, font, (255, 255, 255))
                    draw_text(game_surface, f'Best Score: {best_score_ever} ({best_score_distance:.0f} dist)', 5, 50, font, (255, 255, 255))
                    draw_text(game_surface, f'Best {fitness_label}: {best_fitness_ever:.0f} ({best_fitness_score} pipes)', 5, 70, font, (255, 200, 100))
                    draw_text(game_surface, 'ESC: Save & Exit', 5, SCREENHEIGHT - 20, font, (255, 255, 0))

                    screen.blit(scale_to_screen(game_surface, screen), (0, 0))
                    pygame.display.update()
                    clock.tick(60)

            # Store distances from this run
            for i in range(len(birds)):
                all_run_distances[i].append(run_distances[i])
                cumulative_scores[i] += birds[i].score

        # After all runs: calculate fitness based on method
        fitness_method = settings.get('fitness_method', 'min')

        for i, bird in enumerate(birds):
            distances = all_run_distances[i]
            if fitness_method == 'avg':
                bird.distance = sum(distances) / len(distances)
            elif fitness_method == 'min':
                bird.distance = min(distances)
            elif fitness_method == 'geo':
                # Geometric mean: (d1 * d2 * ... * dn)^(1/n)
                product = 1.0
                for d in distances:
                    product *= max(d, 1)  # Avoid zero
                bird.distance = product ** (1.0 / len(distances))
            elif fitness_method == 'harm':
                # Harmonic mean: n / (1/d1 + 1/d2 + ... + 1/dn)
                inv_sum = sum(1.0 / max(d, 1) for d in distances)
                bird.distance = len(distances) / inv_sum if inv_sum > 0 else 0

        # Calculate stats for logging
        distances = [bird.distance for bird in birds]
        mean_dist = sum(distances) / len(distances)
        sigma = (sum((d - mean_dist) ** 2 for d in distances) / len(distances)) ** 0.5
        best_dist = max(distances)

        # Find best bird this generation (by fitness)
        best_bird_idx = distances.index(best_dist)
        best_bird_score = max(cumulative_scores[best_bird_idx] // runs_per_bird, gen_best_score)

        # Update bests
        if gen_best_score > best_score_ever:
            best_score_ever = gen_best_score
            # Find the bird that got this score and get their fitness
            best_score_distance = best_dist  # Approximate with best fitness bird

        if best_dist > best_fitness_ever:
            best_fitness_ever = best_dist
            best_fitness_score = best_bird_score

        # Log this generation
        training_log.append({
            'generation': generation,
            'best_distance': best_dist,
            'mean_distance': mean_dist,
            'sigma': sigma,
            'best_score': gen_best_score,
        })

        fitness_label = FITNESS_LABELS.get(fitness_method, fitness_method)
        print(f"Gen {generation} | Score: {gen_best_score} | Best: {best_score_ever} pipes/{best_score_distance:.0f} dist | {fitness_label}: {best_fitness_ever:.0f} dist/{best_fitness_score} pipes | Mean: {mean_dist:.0f} | σ: {sigma:.0f}")

        # Evolve population
        population.evolve()
        generation += 1


def getRandomPipe():
    """Returns a randomly generated pipe."""
    gapY = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gapY += int(BASEY * 0.2)
    pipeHeight = IMAGES['pipe'][0].get_height()
    pipeX = SCREENWIDTH + 10
    return [
        {'x': pipeX, 'y': gapY - pipeHeight},
        {'x': pipeX, 'y': gapY + PIPEGAPSIZE},
    ]


def showScore(surface, score):
    """Display score on surface."""
    scoreDigits = [int(x) for x in list(str(score))]
    totalWidth = sum(IMAGES['numbers'][d].get_width() for d in scoreDigits)
    Xoffset = (SCREENWIDTH - totalWidth) / 2
    for digit in scoreDigits:
        surface.blit(IMAGES['numbers'][digit], (Xoffset, SCREENHEIGHT * 0.1))
        Xoffset += IMAGES['numbers'][digit].get_width()


def checkCrash(player, upperPipes, lowerPipes):
    """Returns True if player collides with base or pipes."""
    pw = IMAGES['player'][0].get_width()
    ph = IMAGES['player'][0].get_height()

    # Ground/ceiling collision
    if player['y'] + ph >= BASEY - 1 or player['y'] < 0:
        return [True, True]

    # Shrink hitbox slightly for more forgiving collisions
    margin = 3
    playerRect = pygame.Rect(player['x'] + margin, player['y'] + margin,
                              pw - margin * 2, ph - margin * 2)

    pipeW = IMAGES['pipe'][0].get_width()
    pipeH = IMAGES['pipe'][0].get_height()

    for uPipe, lPipe in zip(upperPipes, lowerPipes):
        uPipeRect = pygame.Rect(uPipe['x'], uPipe['y'], pipeW, pipeH)
        lPipeRect = pygame.Rect(lPipe['x'], lPipe['y'], pipeW, pipeH)

        if playerRect.colliderect(uPipeRect) or playerRect.colliderect(lPipeRect):
            return [True, False]

    return [False, False]


def getHitmask(image):
    """Returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


def main():
    pygame.init()

    # Resizable window, starts at 2x size, with double buffering
    screen = pygame.display.set_mode((SCREENWIDTH * 2, SCREENHEIGHT * 2), pygame.RESIZABLE | pygame.DOUBLEBUF)
    # Fixed-size render surface (game logic always uses this resolution)
    game_surface = pygame.Surface((SCREENWIDTH, SCREENHEIGHT))
    pygame.display.set_caption('Flappy Bird ML')

    # Fonts
    font = pygame.font.Font(None, 24)
    font_large = pygame.font.Font(None, 36)

    # Load assets
    load_assets()
    load_random_sprites()

    # Game state machine
    state = STATE_MENU
    model_name = None
    pending_mode = None
    train_settings = None

    while True:
        if state == STATE_MENU:
            state, pending_mode = main_menu(screen, game_surface, font, font_large)

        elif state == STATE_MODEL_SELECT:
            state, model_name = model_select_menu(screen, game_surface, font, font_large, pending_mode)
            if state == STATE_MENU:
                model_name = None

        elif state == STATE_TRAIN_SETTINGS:
            state, train_settings = train_settings_menu(screen, game_surface, font, font_large, model_name)
            if state == STATE_MENU:
                model_name = None
                train_settings = None

        elif state == STATE_PLAY:
            state = play_game(screen, game_surface, font)
            load_random_sprites()

        elif state == STATE_RUN_MODEL:
            state = run_model_game(screen, game_surface, font, model_name)
            model_name = None
            load_random_sprites()

        elif state == STATE_TRAIN:
            state = train_game(screen, game_surface, font, font_large, model_name, train_settings)
            model_name = None
            train_settings = None
            load_random_sprites()


if __name__ == '__main__':
    main()
