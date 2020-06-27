class OmniTask:
    def __init__(self, K, N, noise_percent):
        # K, N as in N-way K-shot learning
        self.K = K
        self.N = N
        self.noise_percent = noise_percent
        self.mini_train = None
        self.mini_test = None
        self.image_dir = "omniglot_dataset/images_background/*/*"
        self.trainTransform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        self.mini_batch_size = 5

    def mini_train_set(self):
        if self.mini_train is None:
            # choose N tasks
            characters = random.sample(glob.glob(self.image_dir), self.N)

            train_set = []
            # choose K examples per task:
            for i, char in enumerate(characters):
                k_shots = random.sample(glob.glob(char + "/*"), self.K)
                train_set.extend([(self.trainTransform(Image.open(shot).convert('RGB')), i) for shot in k_shots])
            self.mini_train = random.sample(train_set, len(train_set))

        return self.mini_train

    def batched_mini_train_set(self):
        train_set = self.mini_train_set()
        shuffled = random.sample(train_set, len(train_set))

        batched = []

        current_xes = []
        current_yes = []
        for i in range(len(shuffled)):
            if (i % self.mini_batch_size == 0 and i > 0) or (i == len(shuffled) - 1):
                batched.append((torch.stack(current_xes), torch.LongTensor(current_yes)))
                current_xes = []
                current_yes = []
            current_xes.append(shuffled[i][0])
            current_yes.append(shuffled[i][1])

        return batched


    def mini_test_set(self):
        #USE ORIGINAL EVAL FUNCTIONS
        pass


    def eval_set(self, size=50):
        pass


class DataGenerator:
    def __init__(self, size=50000, K=5, N=5, noise_percent=0):
        self.size = size
        self.K = K
        self.N = N
        self.noise_percent = noise_percent
        self.tasks = None

    def generate_set(self):
        self.tasks = tasks = [OmniTask(self.K, self.N, self.noise_percent) for _ in range(self.size)]
        return tasks

    def shuffled_set(self):
        if self.tasks is None:
            self.generate_set()
        return random.sample(self.tasks, len(self.tasks))

class OmniglotNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.conv = nn.Sequential(
            # 28 x 28 - 1
            nn.Conv2d(1, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 14 x 14 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 7 x 7 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 4 x 4 - 64
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 2 x 2 - 64
        )

        self.classifier = nn.Sequential(
            # 2 x 2 x 64 = 256
            nn.Linear(256, num_classes),
            nn.LogSoftmax(1)
        )

    def forward(self, x):
        out = x.view(-1, 1, 28, 28)
        out = self.conv(out)
        out = out.view(len(out), -1) # should be equivalent to out.view(-1, 256)
        out = self.classifier(out)
        return out

    def predict(self, prob):
        __, argmax = prob.max(1)
        return argmax