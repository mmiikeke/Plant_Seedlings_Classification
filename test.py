CUDA_DEVICES = 0

one_hot_key = {}

def test():
    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    root_dir = "plant-seedlings-classification/train"
    i = 0
    for f in listdir(root_dir):
        one_hot_key[i] = f
        i += 1
    
    model = torch.load('model-weight_and_bias.pth')
    model = model.cuda(CUDA_DEVICES)
    model.eval()

    sample_submission = pd.read_csv('plant-seedlings-classification/sample_submission.csv')
    submission = sample_submission.copy()
    
    for i, filename in enumerate(sample_submission['file']):
        print(filename)
        image = Image.open(join('plant-seedlings-classification/test', filename)).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        inputs = Variable(image.cuda(CUDA_DEVICES))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        
        print(preds[0].item())
        submission['species'][i] = one_hot_key[preds[0].item()]

    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    test()
