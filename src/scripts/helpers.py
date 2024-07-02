from torchvision import transforms

def train_transform():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform


def test_transform():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform
