import torchvision.transforms as transforms

# Define more diverse TTA transformations
tta_transforms = [
    # Normal test transform
    transforms.Compose([
        transforms.ToTensor(),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
    ]),
    # Color jitter 1
    transforms.Compose([
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ]),
    # Random crop
    transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ]),
    # Color jitter 2 (different parameters)
    transforms.Compose([
        transforms.ColorJitter(saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ]),
    # Rotation +5 degrees
    transforms.Compose([
        transforms.RandomRotation(degrees=(5, 5)),
        transforms.ToTensor(),
    ]),
    # Rotation -5 degrees
    transforms.Compose([
        transforms.RandomRotation(degrees=(-5, -5)),
        transforms.ToTensor(),
    ]),
    # Zoom in (center crop + resize)
    transforms.Compose([
        transforms.CenterCrop(28),
        transforms.Resize(32),
        transforms.ToTensor(),
    ]),
    # Brightness adjustment
    transforms.Compose([
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
    ]),
    # Contrast adjustment
    transforms.Compose([
        transforms.ColorJitter(contrast=0.2),
        transforms.ToTensor(),
    ])
]

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


