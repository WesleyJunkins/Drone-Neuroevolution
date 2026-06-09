import os
import shutil


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}: {e}')
        print(f'Cleared: {folder_path}/')
    else:
        print(f'Does not exist (skipped): {folder_path}/')


def clear_all():
    print('Clearing all NE experiment output directories...')
    print('=' * 60)
    for folder in [
        'train_ycommand',
        'train_ycontinuous',
        'train_ymotors',
        'notrain_ycommand',
        'notrain_ycontinuous',
        'notrain_ymotors',
    ]:
        clear_folder(folder)
    print('=' * 60)
    print('Done.')


if __name__ == '__main__':
    clear_all()
