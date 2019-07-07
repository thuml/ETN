from data import *
from net import *
import datetime
from tqdm import tqdm
if is_in_notebook():
    from tqdm import tqdm_notebook as tqdm
from torch import optim
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.deterministic = True


def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything()

if args.misc.gpus < 1:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    gpu_ids = []
    output_device = torch.device('cpu')
else:
    gpu_ids = select_GPUs(args.misc.gpus)
    output_device = gpu_ids[0]

now = datetime.datetime.now().strftime('%b%d_%H-%M-%S')

log_dir = f'{args.log.root_dir}/{now}'

logger = SummaryWriter(log_dir)

with open(join(log_dir, 'config.yaml'), 'w') as f:
    f.write(yaml.dump(save_config))

log_text = open(join(log_dir, 'log.txt'), 'w')

model_dict = {
    'resnet50': ResNet50Fc,
    'vgg16': VGG16Fc
}


class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.feature_extractor = model_dict[args.model.base_model](args.model.pretrained_model)
        classifier_output_dim = len(source_classes)
        self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=256)
        self.discriminator = AdversarialNetwork(256)
        self.classifier_auxiliary = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, classifier_output_dim),
            TorchLeakySoftmax(classifier_output_dim)
        )

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        y_aug, d_aug = self.classifier_auxiliary(_)
        return y, d, y_aug, d_aug

totalNet = TotalNet()

logger.add_graph(totalNet, torch.ones(2, 3, 224, 224))
feature_extractor = nn.DataParallel(totalNet.feature_extractor, device_ids=gpu_ids, output_device=output_device).train(True)
classifier = nn.DataParallel(totalNet.classifier, device_ids=gpu_ids, output_device=output_device).train(True)
discriminator = nn.DataParallel(totalNet.discriminator, device_ids=gpu_ids, output_device=output_device).train(True)
classifier_auxiliary = nn.DataParallel(totalNet.classifier_auxiliary, device_ids=gpu_ids, output_device=output_device).train(True)

if args.test.test_only:
    assert os.path.exists(args.test.resume_file)
    data = torch.load(open(args.test.resume_file, 'rb'))
    feature_extractor.load_state_dict(data['feature_extractor'])
    classifier.load_state_dict(data['classifier'])
    discriminator.load_state_dict(data['discriminator'])
    classifier_auxiliary.load_state_dict(data['classifier_auxiliary'])

    counter = AccuracyCounter()
    with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():
        for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
            im = im.to(output_device)
            label = label.to(output_device)

            feature = feature_extractor.forward(im)
            ___, __, before_softmax, predict_prob = classifier.forward(feature)

            counter.addOneBatch(variable_to_numpy(predict_prob),
                                variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

    acc_test = counter.reportAccuracy()
    print(f'test accuracy is {acc_test}')
    exit(0)


# ===================optimizer
scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=10000)
optimizer_finetune = OptimWithSheduler(
    optim.SGD(feature_extractor.parameters(), lr=args.train.lr / 10.0, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_cls = OptimWithSheduler(
    optim.SGD(classifier.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_discriminator = OptimWithSheduler(
    optim.SGD(discriminator.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)
optimizer_classifier_auxiliary = OptimWithSheduler(
    optim.SGD(classifier_auxiliary.parameters(), lr=args.train.lr, weight_decay=args.train.weight_decay, momentum=args.train.momentum, nesterov=True),
    scheduler)

global_step = 0
best_acc = 0

total_steps = tqdm(range(args.train.min_step),desc='global step')
epoch_id = 0

while global_step < args.train.min_step:

    iters = tqdm(zip(source_train_dl, target_train_dl), desc=f'epoch {epoch_id} ', total=min(len(source_train_dl), len(target_train_dl)))
    epoch_id += 1

    for i, ((im_source, label_source), (im_target, label_target)) in enumerate(iters):

        save_label_target = label_target  # for debug usage

        label_source = label_source.to(output_device)
        label_target = label_target.to(output_device)
        label_target = torch.zeros_like(label_target)

        # =========================forward pass
        im_source = im_source.to(output_device)
        im_target = im_target.to(output_device)

        fc1_s = feature_extractor.forward(im_source)
        fc1_t = feature_extractor.forward(im_target)

        fc1_s, feature_source, fc2_s, predict_prob_source = classifier.forward(fc1_s)
        fc1_t, feature_target, fc2_t, predict_prob_target = classifier.forward(fc1_t)

        domain_prob_discriminator_source = discriminator.forward(feature_source)
        domain_prob_discriminator_target = discriminator.forward(feature_target)

        predict_prob_source_aug, domain_prob_source_aug = classifier_auxiliary.forward(feature_source.detach())
        predict_prob_target_aug, domain_prob_target_aug = classifier_auxiliary.forward(feature_target.detach())

        # ==============================compute loss
        weight = (1.0 - domain_prob_source_aug)
        weight = weight / (torch.mean(weight, dim=0, keepdim=True) + 1e-10)
        weight = weight.detach()

        # ============================== cross entropy loss, it receives logits as its inputs
        ce = nn.CrossEntropyLoss(reduction='none')(fc2_s, label_source).view(-1, 1)
        ce = torch.mean(ce * weight, dim=0, keepdim=True)

        tmp = weight * nn.BCELoss(reduction='none')(domain_prob_discriminator_source, torch.ones_like(domain_prob_discriminator_source))
        adv_loss = torch.mean(tmp, dim=0, keepdim=True)
        adv_loss += nn.BCELoss()(domain_prob_discriminator_target, torch.zeros_like(domain_prob_discriminator_target))

        ce_aug = nn.BCELoss(reduction='none')(predict_prob_source_aug, one_hot(label_source, args.data.dataset.n_total))
        ce_aug = torch.sum(ce_aug) / label_source.numel()
        adv_loss_aug = nn.BCELoss()(domain_prob_source_aug, torch.ones_like(domain_prob_source_aug))
        adv_loss_aug += nn.BCELoss()(domain_prob_target_aug, torch.zeros_like(domain_prob_target_aug))

        entropy = EntropyLoss(predict_prob_target)

        with OptimizerManager(
                [optimizer_finetune, optimizer_cls, optimizer_discriminator, optimizer_classifier_auxiliary]):
            loss = ce + args.train.adv_loss_tradeoff * adv_loss + args.train.entropy_tradeoff * entropy + \
                   args.train.adv_loss_aug_tradeoff * adv_loss_aug + args.train.ce_aug_tradeoff * ce_aug
            loss.backward()

        global_step += 1
        total_steps.update()

        if global_step % args.log.log_interval == 0:
            counter = AccuracyCounter()
            counter.addOneBatch(variable_to_numpy(one_hot(label_source, len(source_classes))), variable_to_numpy(predict_prob_source))
            acc_train = torch.tensor([counter.reportAccuracy()]).to(output_device)
            logger.add_scalar('entropy', entropy, global_step)
            logger.add_scalar('adv_loss', adv_loss, global_step)
            logger.add_scalar('ce', ce, global_step)
            logger.add_scalar('adv_loss_aug', adv_loss_aug, global_step)
            logger.add_scalar('ce_aug', ce_aug, global_step)
            logger.add_scalar('acc_train', acc_train, global_step)

        if global_step % args.test.test_interval == 0:

            counter = AccuracyCounter()
            with TrainingModeManager([feature_extractor, classifier], train=False) as mgr, torch.no_grad():

                for i, (im, label) in enumerate(tqdm(target_test_dl, desc='testing ')):
                    im = im.to(output_device)
                    label = label.to(output_device)

                    feature = feature_extractor.forward(im)
                    ___, __, before_softmax, predict_prob = classifier.forward(feature)

                    counter.addOneBatch(variable_to_numpy(predict_prob), variable_to_numpy(one_hot(label, args.data.dataset.n_total)))

            acc_test = counter.reportAccuracy()
            logger.add_scalar('acc_test', acc_test, global_step)
            log_text.write(f'acc_test is {acc_test} \n')
            log_text.flush()
            clear_output()

            data = {
                "feature_extractor": feature_extractor.state_dict(),
                'classifier': classifier.state_dict(),
                'discriminator': discriminator.state_dict(),
                'classifier_auxiliary': classifier_auxiliary.state_dict(),
            }

            if acc_test > best_acc:
                best_acc = acc_test
                with open(join(log_dir, 'best.pkl'), 'wb') as f:
                    torch.save(data, f)

            with open(join(log_dir, 'current.pkl'), 'wb') as f:
                torch.save(data, f)
