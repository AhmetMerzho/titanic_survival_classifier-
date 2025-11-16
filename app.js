// Titanic Survival Classifier (TensorFlow.js)
// ------------------------------------------------------------
// This file contains all browser-side logic: loading CSV files,
// preprocessing, model creation/training, evaluation, prediction
// and exporting artifacts.

let trainData = [];
let testData = [];
let preprocessedTrainData = null;
let preprocessedTestData = null;
let preprocessingStats = null;
let model = null;
let validationProbabilities = [];
let validationLabelsArray = [];
let testPredictionProbs = [];
let rocCache = null;

const TARGET_FEATURE = 'Survived';
const ID_FEATURE = 'PassengerId';
const NUMERICAL_FEATURES = ['Age', 'Fare', 'SibSp', 'Parch'];
const CATEGORICAL_FEATURES = ['Pclass', 'Sex', 'Embarked'];
const FIXED_CATEGORIES = {
    Pclass: [1, 2, 3],
    Sex: ['male', 'female'],
    Embarked: ['C', 'Q', 'S']
};

function disableWorkflowButtons() {
    document.getElementById('inspect-btn').disabled = true;
    document.getElementById('preprocess-btn').disabled = true;
    document.getElementById('create-model-btn').disabled = true;
    document.getElementById('train-btn').disabled = true;
    document.getElementById('predict-btn').disabled = true;
    document.getElementById('export-btn').disabled = true;
    const slider = document.getElementById('threshold-slider');
    slider.disabled = true;
    slider.value = 0.5;
    document.getElementById('threshold-value').textContent = '0.50';
}

disableWorkflowButtons();

// ---------- Data Loading ----------
async function loadData() {
    const trainFile = document.getElementById('train-file').files[0];
    const testFile = document.getElementById('test-file').files[0];

    if (!trainFile || !testFile) {
        alert('Please upload both training and test CSV files.');
        return;
    }

    disableWorkflowButtons();
    const statusDiv = document.getElementById('data-status');
    statusDiv.innerHTML = 'Loading data...';

    try {
        const [trainText, testText] = await Promise.all([readFile(trainFile), readFile(testFile)]);
        trainData = parseCSV(trainText);
        testData = parseCSV(testText);

        validateColumns(trainData, true);
        validateColumns(testData, false);

        statusDiv.innerHTML = `Data loaded successfully! Training: ${trainData.length} samples, Test: ${testData.length} samples.`;

        document.getElementById('inspect-btn').disabled = false;
        clearDownstreamOutputs();
    } catch (error) {
        statusDiv.innerHTML = `Error loading data: ${error.message}`;
        console.error(error);
    }
}

function readFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = e => resolve(e.target.result);
        reader.onerror = () => reject(new Error('Failed to read file.'));
        reader.readAsText(file);
    });
}

function parseCSV(csvText) {
    if (!window.Papa) {
        throw new Error('PapaParse library is not loaded.');
    }

    const numericColumns = new Set([
        TARGET_FEATURE,
        ID_FEATURE,
        ...NUMERICAL_FEATURES,
        ...CATEGORICAL_FEATURES,
    ]);

    const result = Papa.parse(csvText.trim(), {
        header: true,
        dynamicTyping: header => numericColumns.has(header),
        skipEmptyLines: 'greedy'
    });

    if (result.errors && result.errors.length > 0) {
        throw new Error(`CSV parse error: ${result.errors[0].message}`);
    }

    return result.data.filter(row => Object.values(row).some(value => value !== null && value !== undefined && value !== ''));
}

function validateColumns(dataset, isTrain) {
    const requiredColumns = new Set([ID_FEATURE, ...NUMERICAL_FEATURES, ...CATEGORICAL_FEATURES]);
    if (isTrain) {
        requiredColumns.add(TARGET_FEATURE);
    }

    if (!dataset || dataset.length === 0) {
        throw new Error('Dataset is empty after parsing.');
    }

    const sampleRow = dataset[0];
    requiredColumns.forEach(column => {
        if (!(column in sampleRow)) {
            throw new Error(`Missing required column "${column}" in ${isTrain ? 'training' : 'test'} data.`);
        }
    });
}

function clearDownstreamOutputs() {
    document.getElementById('data-preview').innerHTML = '';
    document.getElementById('data-stats').innerHTML = '';
    document.getElementById('charts').innerHTML = '';
    document.getElementById('preprocessing-output').innerHTML = '';
    document.getElementById('model-summary').innerHTML = '';
    document.getElementById('training-status').innerHTML = '';
    document.getElementById('prediction-output').innerHTML = '';
    document.getElementById('export-status').innerHTML = '';
    document.getElementById('confusion-matrix').innerHTML = '';
    document.getElementById('performance-metrics').innerHTML = '';
    validationProbabilities = [];
    validationLabelsArray = [];
    rocCache = null;
    testPredictionProbs = [];
    preprocessedTrainData?.featuresTensor?.dispose?.();
    preprocessedTrainData?.labelsTensor?.dispose?.();
    preprocessedTrainData = null;
    preprocessedTestData = null;
    preprocessingStats = null;
    if (model) {
        model.dispose();
    }
    model = null;
    const slider = document.getElementById('threshold-slider');
    slider.disabled = true;
    slider.value = 0.5;
    document.getElementById('threshold-value').textContent = '0.50';
}

// ---------- Inspection ----------
function inspectData() {
    if (!trainData.length) {
        alert('Please load data first.');
        return;
    }

    const previewDiv = document.getElementById('data-preview');
    previewDiv.innerHTML = '<h3>Data Preview (First 10 Rows)</h3>';
    previewDiv.appendChild(createPreviewTable(trainData.slice(0, 10)));

    const statsDiv = document.getElementById('data-stats');
    const shapeInfo = `Dataset shape: ${trainData.length} rows x ${Object.keys(trainData[0]).length} columns`;
    const survivalCount = trainData.filter(row => Number(row[TARGET_FEATURE]) === 1).length;
    const survivalRate = (survivalCount / trainData.length * 100).toFixed(2);
    const targetInfo = `Survival rate: ${survivalCount}/${trainData.length} (${survivalRate}%)`;

    let missingInfo = '<h4>Missing Values Percentage:</h4><ul>';
    Object.keys(trainData[0]).forEach(feature => {
        const missingCount = trainData.filter(row => row[feature] === null || row[feature] === undefined || row[feature] === '').length;
        missingInfo += `<li>${feature}: ${(missingCount / trainData.length * 100).toFixed(2)}%</li>`;
    });
    missingInfo += '</ul>';

    statsDiv.innerHTML = `<h3>Data Statistics</h3><p>${shapeInfo}</p><p>${targetInfo}</p>${missingInfo}`;

    createVisualizations();
    document.getElementById('preprocess-btn').disabled = false;
}

function createPreviewTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    Object.keys(data[0]).forEach(key => {
        const th = document.createElement('th');
        th.textContent = key;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach(row => {
        const tr = document.createElement('tr');
        Object.values(row).forEach(value => {
            const td = document.createElement('td');
            td.textContent = value !== null && value !== undefined ? value : 'NULL';
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

function createVisualizations() {
    const chartsDiv = document.getElementById('charts');
    chartsDiv.innerHTML = '<h3>Data Visualizations</h3><p>Charts are displayed in the tfjs-vis visor (bottom-right button).</p>';

    const survivalBySex = {};
    trainData.forEach(row => {
        const sex = row.Sex !== undefined && row.Sex !== null ? row.Sex.toString() : '';
        if (sex === '') return;
        if (!survivalBySex[sex]) {
            survivalBySex[sex] = { survived: 0, total: 0 };
        }
        survivalBySex[sex].total += 1;
        if (Number(row[TARGET_FEATURE]) === 1) {
            survivalBySex[sex].survived += 1;
        }
    });
    const sexChart = Object.entries(survivalBySex).map(([sex, stats]) => ({
        x: sex,
        y: stats.total ? (stats.survived / stats.total) * 100 : 0
    }));

    tfvis.render.barchart({ name: 'Survival Rate by Sex', tab: 'Inspection' }, sexChart, {
        xLabel: 'Sex',
        yLabel: 'Survival Rate (%)'
    });

    const survivalByClass = {};
    trainData.forEach(row => {
        const pclass = row.Pclass;
        if (pclass === null || pclass === undefined) return;
        if (!survivalByClass[pclass]) {
            survivalByClass[pclass] = { survived: 0, total: 0 };
        }
        survivalByClass[pclass].total += 1;
        if (Number(row[TARGET_FEATURE]) === 1) {
            survivalByClass[pclass].survived += 1;
        }
    });
    const classChart = Object.entries(survivalByClass).map(([klass, stats]) => ({
        x: `Class ${klass}`,
        y: stats.total ? (stats.survived / stats.total) * 100 : 0
    }));

    tfvis.render.barchart({ name: 'Survival Rate by Passenger Class', tab: 'Inspection' }, classChart, {
        xLabel: 'Passenger Class',
        yLabel: 'Survival Rate (%)'
    });
}

// ---------- Preprocessing ----------
function preprocessData() {
    if (!trainData.length || !testData.length) {
        alert('Please load and inspect data first.');
        return;
    }

    const outputDiv = document.getElementById('preprocessing-output');
    outputDiv.innerHTML = 'Preprocessing data...';

    try {
        preprocessingStats = computePreprocessingStats(trainData);
        preprocessingStats.includeFamily = document.getElementById('add-family-features').checked;

        const trainFeaturesArray = [];
        const trainLabelsArray = [];

        trainData.forEach((row, index) => {
            if (row[TARGET_FEATURE] === undefined || row[TARGET_FEATURE] === null || row[TARGET_FEATURE] === '') {
                throw new Error(`Missing target value on row ${index + 2}.`);
            }
            trainFeaturesArray.push(extractFeatures(row, preprocessingStats));
            trainLabelsArray.push(Number(row[TARGET_FEATURE]));
        });

        const testFeaturesArray = [];
        const testPassengerIds = [];
        testData.forEach(row => {
            testFeaturesArray.push(extractFeatures(row, preprocessingStats));
            testPassengerIds.push(row[ID_FEATURE]);
        });

        preprocessedTrainData?.featuresTensor?.dispose?.();
        preprocessedTrainData?.labelsTensor?.dispose?.();

        preprocessedTrainData = {
            featuresArray: trainFeaturesArray,
            labelsArray: trainLabelsArray,
            featuresTensor: tf.tensor2d(trainFeaturesArray),
            labelsTensor: tf.tensor1d(trainLabelsArray)
        };

        preprocessedTestData = {
            featuresArray: testFeaturesArray,
            passengerIds: testPassengerIds
        };

        if (model) {
            model.dispose();
        }
        model = null;
        validationProbabilities = [];
        validationLabelsArray = [];
        rocCache = null;
        testPredictionProbs = [];
        document.getElementById('model-summary').innerHTML = '';
        document.getElementById('training-status').innerHTML = '';
        document.getElementById('confusion-matrix').innerHTML = '';
        document.getElementById('performance-metrics').innerHTML = '';
        document.getElementById('prediction-output').innerHTML = '';
        document.getElementById('export-status').innerHTML = '';
        document.getElementById('export-btn').disabled = true;
        document.getElementById('predict-btn').disabled = true;
        document.getElementById('train-btn').disabled = true;
        const slider = document.getElementById('threshold-slider');
        slider.disabled = true;
        slider.value = 0.5;
        document.getElementById('threshold-value').textContent = '0.50';

        outputDiv.innerHTML = `
            <p>Preprocessing completed!</p>
            <p>Training features shape: [${preprocessedTrainData.featuresTensor.shape.join(', ')}]</p>
            <p>Training labels shape: [${preprocessedTrainData.labelsTensor.shape.join(', ')}]</p>
            <p>Test features shape: [${preprocessedTestData.featuresArray.length}, ${testFeaturesArray[0]?.length || 0}]</p>
            <p>Imputation → Age/Fare median, Embarked mode. Standardization → Age/Fare mean & std from training data only.</p>
        `;

        document.getElementById('create-model-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during preprocessing: ${error.message}`;
        console.error(error);
    }
}

function computePreprocessingStats(dataset) {
    const ageValues = dataset.map(row => toNumber(row.Age)).filter(isFiniteNumber);
    const fareValues = dataset.map(row => toNumber(row.Fare)).filter(isFiniteNumber);
    const embarkedValues = dataset
        .map(row => (row.Embarked !== null && row.Embarked !== undefined ? row.Embarked.toString().trim().toUpperCase() : ''))
        .filter(val => val !== '');

    return {
        ageMedian: calculateMedian(ageValues),
        ageMean: calculateMean(ageValues),
        ageStd: calculateStd(ageValues),
        fareMedian: calculateMedian(fareValues),
        fareMean: calculateMean(fareValues),
        fareStd: calculateStd(fareValues),
        embarkedMode: calculateMode(embarkedValues) || 'S'
    };
}

function extractFeatures(row, stats) {
    const age = isFiniteNumber(row.Age) ? Number(row.Age) : stats.ageMedian;
    const fare = isFiniteNumber(row.Fare) ? Number(row.Fare) : stats.fareMedian;
    const sibSp = isFiniteNumber(row.SibSp) ? Number(row.SibSp) : 0;
    const parch = isFiniteNumber(row.Parch) ? Number(row.Parch) : 0;
    const embarkedRaw = row.Embarked && row.Embarked !== '' ? row.Embarked.toString().trim().toUpperCase() : stats.embarkedMode;
    const sexRaw = row.Sex ? row.Sex.toString().trim().toLowerCase() : 'male';
    const pclassRaw = isFiniteNumber(row.Pclass) ? Number(row.Pclass) : 3;

    const standardizedAge = stats.ageStd ? (age - stats.ageMean) / stats.ageStd : 0;
    const standardizedFare = stats.fareStd ? (fare - stats.fareMean) / stats.fareStd : 0;

    const pclassOneHot = oneHotEncode(pclassRaw, FIXED_CATEGORIES.Pclass);
    const sexOneHot = oneHotEncode(sexRaw, FIXED_CATEGORIES.Sex);
    const embarkedOneHot = oneHotEncode(embarkedRaw, FIXED_CATEGORIES.Embarked);

    const features = [standardizedAge, standardizedFare, sibSp, parch, ...pclassOneHot, ...sexOneHot, ...embarkedOneHot];

    if (stats.includeFamily) {
        const familySize = sibSp + parch + 1;
        const isAlone = familySize === 1 ? 1 : 0;
        features.push(familySize, isAlone);
    }

    return features;
}

function oneHotEncode(value, categories) {
    const encoding = new Array(categories.length).fill(0);
    const index = categories.indexOf(value);
    if (index >= 0) {
        encoding[index] = 1;
    }
    return encoding;
}

function toNumber(value) {
    if (value === null || value === undefined || value === '') return NaN;
    return Number(value);
}

function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

function calculateMedian(values) {
    if (!values.length) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const half = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[half - 1] + sorted[half]) / 2 : sorted[half];
}

function calculateMean(values) {
    if (!values.length) return 0;
    const sum = values.reduce((acc, val) => acc + val, 0);
    return sum / values.length;
}

function calculateStd(values) {
    if (!values.length) return 0;
    const mean = calculateMean(values);
    const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance) || 0;
}

function calculateMode(values) {
    if (!values.length) return null;
    const freq = {};
    let best = values[0];
    let bestCount = 0;
    values.forEach(val => {
        freq[val] = (freq[val] || 0) + 1;
        if (freq[val] > bestCount) {
            best = val;
            bestCount = freq[val];
        }
    });
    return best;
}

// ---------- Model ----------
function createModel() {
    if (!preprocessedTrainData) {
        alert('Please preprocess data first.');
        return;
    }

    model = tf.sequential();
    const inputShape = preprocessedTrainData.featuresTensor.shape[1];

    model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [inputShape] }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

    const summaryDiv = document.getElementById('model-summary');
    summaryDiv.innerHTML = '<h3>Model Summary</h3>';
    let summaryText = '<ul>';
    model.layers.forEach((layer, i) => {
        summaryText += `<li>Layer ${i + 1}: ${layer.getClassName()} - Output Shape: ${JSON.stringify(layer.outputShape)}</li>`;
    });
    summaryText += '</ul>';
    summaryText += `<p>Total parameters: ${model.countParams()}</p>`;
    summaryDiv.innerHTML += summaryText;

    document.getElementById('train-btn').disabled = false;
}

// ---------- Training ----------
async function trainModel() {
    if (!model || !preprocessedTrainData) {
        alert('Please create the model after preprocessing.');
        return;
    }

    const statusDiv = document.getElementById('training-status');
    statusDiv.innerHTML = 'Training model...';

    try {
        const split = stratifiedSplit(preprocessedTrainData.featuresArray, preprocessedTrainData.labelsArray, 0.2);
        if (!split.trainFeatures.length || !split.valFeatures.length) {
            throw new Error('Not enough data to create stratified train/validation splits.');
        }
        const trainFeatures = tf.tensor2d(split.trainFeatures);
        const trainLabels = tf.tensor1d(split.trainLabels);
        const valFeatures = tf.tensor2d(split.valFeatures);
        const valLabels = tf.tensor1d(split.valLabels);

        const visCallbacks = tfvis.show.fitCallbacks({ name: 'Training Performance', tab: 'Training' }, ['loss', 'val_loss', 'acc', 'val_acc'], { callbacks: ['onEpochEnd'] });
        const earlyStop = tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 5, restoreBestWeight: true });
        const logCallback = {
            onEpochEnd: (epoch, logs) => {
                if (logs.acc === undefined && logs.accuracy !== undefined) {
                    logs.acc = logs.accuracy;
                }
                if (logs.val_acc === undefined && logs.val_accuracy !== undefined) {
                    logs.val_acc = logs.val_accuracy;
                }
                const acc = logs.acc ?? 0;
                const valAcc = logs.val_acc ?? 0;
                statusDiv.innerHTML = `Epoch ${epoch + 1}/50 - loss: ${logs.loss.toFixed(4)}, acc: ${acc.toFixed(4)}, val_loss: ${logs.val_loss.toFixed(4)}, val_acc: ${valAcc.toFixed(4)}`;
            }
        };

        const callbacks = [logCallback, visCallbacks, earlyStop];

        await model.fit(trainFeatures, trainLabels, {
            epochs: 50,
            batchSize: 32,
            validationData: [valFeatures, valLabels],
            callbacks
        });

        statusDiv.innerHTML += '<p>Training completed!</p>';

        const predictionTensor = model.predict(valFeatures);
        validationProbabilities = Array.from(predictionTensor.dataSync());
        validationLabelsArray = split.valLabels.slice();
        rocCache = null;
        predictionTensor.dispose();

        valFeatures.dispose();
        valLabels.dispose();
        trainFeatures.dispose();
        trainLabels.dispose();

        const slider = document.getElementById('threshold-slider');
        slider.disabled = false;
        slider.oninput = updateMetrics;
        slider.value = 0.5;
        document.getElementById('threshold-value').textContent = '0.50';
        updateMetrics();

        document.getElementById('predict-btn').disabled = false;
    } catch (error) {
        statusDiv.innerHTML = `Error during training: ${error.message}`;
        console.error(error);
    }
}

function stratifiedSplit(features, labels, valFraction) {
    const posIdx = [];
    const negIdx = [];
    labels.forEach((label, index) => {
        if (Number(label) === 1) posIdx.push(index);
        else negIdx.push(index);
    });

    shuffleArray(posIdx);
    shuffleArray(negIdx);

    const valPosCount = posIdx.length ? Math.max(1, Math.round(posIdx.length * valFraction)) : 0;
    const valNegCount = negIdx.length ? Math.max(1, Math.round(negIdx.length * valFraction)) : 0;

    const valIndices = new Set([...posIdx.slice(0, valPosCount), ...negIdx.slice(0, valNegCount)]);

    const trainFeatures = [];
    const trainLabels = [];
    const valFeatures = [];
    const valLabels = [];

    labels.forEach((label, index) => {
        if (valIndices.has(index)) {
            valFeatures.push(features[index]);
            valLabels.push(label);
        } else {
            trainFeatures.push(features[index]);
            trainLabels.push(label);
        }
    });

    return { trainFeatures, trainLabels, valFeatures, valLabels };
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// ---------- Metrics & Evaluation ----------
function updateMetrics() {
    if (!validationProbabilities.length || !validationLabelsArray.length) return;

    const slider = document.getElementById('threshold-slider');
    const threshold = parseFloat(slider.value);
    document.getElementById('threshold-value').textContent = threshold.toFixed(2);

    let tp = 0, tn = 0, fp = 0, fn = 0;
    for (let i = 0; i < validationProbabilities.length; i++) {
        const prob = validationProbabilities[i];
        const prediction = prob >= threshold ? 1 : 0;
        const actual = validationLabelsArray[i];

        if (prediction === 1 && actual === 1) tp++;
        else if (prediction === 0 && actual === 0) tn++;
        else if (prediction === 1 && actual === 0) fp++;
        else fn++;
    }

    const cmDiv = document.getElementById('confusion-matrix');
    cmDiv.innerHTML = `
        <table>
            <tr><th></th><th>Predicted Positive</th><th>Predicted Negative</th></tr>
            <tr><th>Actual Positive</th><td>${tp}</td><td>${fn}</td></tr>
            <tr><th>Actual Negative</th><td>${fp}</td><td>${tn}</td></tr>
        </table>
    `;

    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    const accuracy = (tp + tn) / (tp + tn + fp + fn);

    const metricsDiv = document.getElementById('performance-metrics');
    const auc = plotROC(validationLabelsArray, validationProbabilities);
    metricsDiv.innerHTML = `
        <p>Accuracy: ${(accuracy * 100).toFixed(2)}%</p>
        <p>Precision: ${precision.toFixed(4)}</p>
        <p>Recall: ${recall.toFixed(4)}</p>
        <p>F1 Score: ${f1.toFixed(4)}</p>
        <p>AUC: ${auc.toFixed(4)}</p>
    `;
}

function plotROC(trueLabels, predictedProbs) {
    if (!rocCache) {
        const thresholds = Array.from({ length: 101 }, (_, i) => i / 100);
        const rocPoints = thresholds.map(threshold => {
            let tp = 0, fp = 0, tn = 0, fn = 0;
            for (let i = 0; i < predictedProbs.length; i++) {
                const prediction = predictedProbs[i] >= threshold ? 1 : 0;
                const actual = trueLabels[i];
                if (actual === 1) {
                    if (prediction === 1) tp++;
                    else fn++;
                } else {
                    if (prediction === 1) fp++;
                    else tn++;
                }
            }
            const tpr = tp + fn === 0 ? 0 : tp / (tp + fn);
            const fpr = fp + tn === 0 ? 0 : fp / (fp + tn);
            return { x: fpr, y: tpr };
        });

        let auc = 0;
        for (let i = 1; i < rocPoints.length; i++) {
            const width = rocPoints[i].x - rocPoints[i - 1].x;
            const height = (rocPoints[i].y + rocPoints[i - 1].y) / 2;
            auc += width * height;
        }

        rocCache = { points: rocPoints, auc };
    }

    tfvis.render.linechart(
        { name: 'ROC Curve', tab: 'Evaluation' },
        { values: [rocCache.points], series: ['ROC Curve'] },
        {
            xLabel: 'False Positive Rate',
            yLabel: 'True Positive Rate',
            yAxisDomain: [0, 1],
            xAxisDomain: [0, 1],
            width: 400,
            height: 400
        }
    );

    return rocCache.auc;
}

// ---------- Prediction & Export ----------
async function predict() {
    if (!model || !preprocessedTestData) {
        alert('Please train the model and preprocess the test data first.');
        return;
    }

    const outputDiv = document.getElementById('prediction-output');
    outputDiv.innerHTML = 'Making predictions...';

    try {
        const testTensor = tf.tensor2d(preprocessedTestData.featuresArray);
        const predictionTensor = model.predict(testTensor);
        testPredictionProbs = Array.from(predictionTensor.dataSync());
        predictionTensor.dispose();
        testTensor.dispose();

        const results = preprocessedTestData.passengerIds.map((id, idx) => ({
            PassengerId: id,
            Survived: testPredictionProbs[idx] >= 0.5 ? 1 : 0,
            Probability: testPredictionProbs[idx]
        }));

        outputDiv.innerHTML = '<h3>Prediction Results (First 10 Rows)</h3>';
        outputDiv.appendChild(createPredictionTable(results.slice(0, 10)));
        outputDiv.innerHTML += `<p>Predictions completed! Total: ${results.length} samples.</p>`;

        document.getElementById('export-btn').disabled = false;
    } catch (error) {
        outputDiv.innerHTML = `Error during prediction: ${error.message}`;
        console.error(error);
    }
}

function createPredictionTable(data) {
    const table = document.createElement('table');
    const headerRow = document.createElement('tr');
    ['PassengerId', 'Survived', 'Probability'].forEach(header => {
        const th = document.createElement('th');
        th.textContent = header;
        headerRow.appendChild(th);
    });
    table.appendChild(headerRow);

    data.forEach(row => {
        const tr = document.createElement('tr');
        ['PassengerId', 'Survived', 'Probability'].forEach(key => {
            const td = document.createElement('td');
            td.textContent = key === 'Probability' ? Number(row[key]).toFixed(4) : row[key];
            tr.appendChild(td);
        });
        table.appendChild(tr);
    });

    return table;
}

async function exportResults() {
    if (!testPredictionProbs.length || !preprocessedTestData) {
        alert('Please generate predictions first.');
        return;
    }

    const statusDiv = document.getElementById('export-status');
    statusDiv.innerHTML = 'Exporting results...';

    try {
        let submissionCSV = 'PassengerId,Survived\n';
        let probabilitiesCSV = 'PassengerId,Probability\n';

        preprocessedTestData.passengerIds.forEach((id, idx) => {
            const probability = testPredictionProbs[idx];
            const survived = probability >= 0.5 ? 1 : 0;
            submissionCSV += `${id},${survived}\n`;
            probabilitiesCSV += `${id},${probability.toFixed(6)}\n`;
        });

        downloadFile('submission.csv', submissionCSV);
        downloadFile('probabilities.csv', probabilitiesCSV);

        await model.save('downloads://titanic-tfjs');

        statusDiv.innerHTML = `
            <p>Export completed!</p>
            <p>Downloaded: submission.csv (PassengerId, Survived)</p>
            <p>Downloaded: probabilities.csv (PassengerId, Probability)</p>
            <p>Model saved to browser downloads.</p>
        `;
    } catch (error) {
        statusDiv.innerHTML = `Error during export: ${error.message}`;
        console.error(error);
    }
}

function downloadFile(filename, content) {
    const blob = new Blob([content], { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
}

// Expose functions to the window for HTML buttons
window.loadData = loadData;
window.inspectData = inspectData;
window.preprocessData = preprocessData;
window.createModel = createModel;
window.trainModel = trainModel;
window.predict = predict;
window.exportResults = exportResults;
