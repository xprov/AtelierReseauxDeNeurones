/**
 * CHALLENGE NEURAL NETWORK
 *
 * Les défis sont des réseaux de neurones avec une ou plusieurs entrées et la
 * ou les valeurs attendues en sortie.
 * 
 * L'utilisateur utilise des curseurs pour modifier la valeur les points
 * associés aux connexions entre les neurones.
 *
 *
 */

function f_act(x) {
  return Math.max(Math.min(x, 100), 0);
}

function createHatchPattern(ctx, color = "black") {
    const patternCanvas = document.createElement("canvas");
    patternCanvas.width = 8;
    patternCanvas.height = 8;

    const pctx = patternCanvas.getContext("2d");

    pctx.strokeStyle = color;
    pctx.lineWidth = 1;

    pctx.beginPath();
    pctx.moveTo(0, 8);
    pctx.lineTo(8, 0);
    pctx.stroke();

    return ctx.createPattern(patternCanvas, "repeat");
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

class Parameter {
  constructor(challengeId, minValue = -1, maxValue = 1.5, defaultValue = 0.25) {
    this.challengeId = challengeId;
    this.minValue = minValue;
    this.maxValue = maxValue;
    this.defaultValue = defaultValue;
    this.label = document.createElement("div");
    this.label.innerHTML = "" + defaultValue;
    this.slider = document.createElement("INPUT");
    this.slider.id = name;
    this.slider.type = "range";
    this.slider.min = this.minValue;
    this.slider.max = this.maxValue;
    this.slider.step = 0.001;
    this.slider.value = this.defaultValue;
    this.slider.style="width: 550px";
    this.slider.setAttribute("challenge", challengeId);
    this.slider.setAttribute("label", this.label);
    this.slider.oninput = function() {Challenge.all[this.getAttribute("challenge")].update()};
  }
}

class Node{

  // L'activation est une valeur entre 0 et 100.
  constructor(x, y, activation)
  {
    this.x = x;
    this.y = y;
    this.activation = activation;
  }
}

class NetworkWithInputs {
  constructor(layers, inputs, expectedOutputs) {
    console.assert(inputs.length == expectedOutputs.length);

    this.numLayers = layers.length;
    this.numCopies = inputs.length;
    this.layersSizes = layers;
    this.inputs = inputs;
    this.expectedOutputs = expectedOutputs;

    // Construction des layers de nodes pour toutes les copies
    this.copies = [];
    for (let copyId=0; copyId<this.numCopies; copyId++) {
      let nodes = [];
      for (let i=0; i<layers.length; i++) {
        let numNodesInThisLayer = layers[i];
        let layer;
        if (i == 0) {
          layer = inputs[copyId];
          console.assert(layer.length == numNodesInThisLayer);
        } else {
          layer = Array(numNodesInThisLayer);
          layer.fill(0);
        }
        nodes.push(layer);
      }
      this.copies.push(nodes);
    }

    // Les poids
    // on peut pouvoir y accéder de deux manières
    // (1) this.weights[ numLayerDeGauche ][ numNeuroneLayerDeGauche ][ numNeuronneLayerDeDroits ]
    // (2) this.weightsIndices --> 0, 1, 2, 3, 4, ...
    this.numWeights = 0;
    this.weights = [];
    this.weightsIndices = [];
    for (let i=0; i<this.numLayers-1; i++) {
      this.weights[i] = [];
      for (let j=0; j<this.layersSizes[i]; j++) {
        this.weights[i][j] = [];
        for (let k=0; k<this.layersSizes[i+1]; k++) {
          this.weights[i][j][k] = 0;
          this.weightsIndices[this.numWeights] = [i, j, k];
          this.numWeights++;
        }
      }
    }
  }

  getWeightIndices(index) {
    return this.weightsIndices[index];
  }

  getWeight(index) {
    let indices = this.weightsIndices[index];
    let i = indices[0];
    let j = indices[1];
    let k = indices[2];
    return this.weights[i][j][k];
  }

  setWeight(index, value) {
    let indices = this.weightsIndices[index];
    let i = indices[0];
    let j = indices[1];
    let k = indices[2];
    this.weights[i][j][k] = value;
  }

  propagate() {
    // Le layer 0 contient les inputs
    for (let nodes of this.copies) {
      for (let i=0; i<this.numLayers-1; i++) {
        for (let k=0; k<this.layersSizes[i+1]; k++) {
          let activation = 0;
          for (let j=0; j<this.layersSizes[i]; j++) {
            activation += this.weights[i][j][k] * nodes[i][j];
          }
          nodes[i+1][k] = f_act(activation);
        }
      }
    }
  }

  // mean square error
  mse() {
    let mse = 0.0;
    let n = 0;
    for (let copyId in this.copies) {
      let obtained = this.copies[copyId][this.numLayers-1];
      let expected = this.expectedOutputs[copyId];
      console.assert(obtained.length == expected.length);
      for (let k in obtained) {
        let err = obtained[k] - expected[k];
        mse += err * err;
        n += 1;
      }
    }
    return mse / n;
  }

}


class Challenge {

  // Dictionnaire contenant tous les challenges. Les clés sont les `challengeId`.
  static all = {};
  static lastAdded = null;

  /**
   * Le `challengeId` doit correcpondre au `id` du DIV dans lequel le challenge est inséré.
   *
   * layers : liste d'entiers, chaque nombre est le nombre de neurones dans une couche.
   * inputs : liste de listes d'entiers. Chaque sous-liste contient des valeurs
   *           pour les neurones de la couche d'entrée.
   * expectedOutputs : liste de listes d'entiers. Chaque sous-liste contient
   *                   les valeurs attendues pour les neurones de la couche de sortie, en
   *                   fonction de l'entrée correspondante du paramètre `inputs`
   * minError : erreur à partir de laquelle le challenge est considéré comme réussi.
   * gradientDescentFactor : facteur multiplicatif sur la taille des pas lors de la desente de gradient.
   * gradientDescentMaxSteps : nombre maximum d'étapes lors de la descente de gradient.
   * epsilon : valeur utilisée pour simuler le calcul de la dérivée.
   */
  constructor(challengeId, layers, inputs, expectedOutputs, 
    minError = 0.001, 
    gradientDescentFactor = 0.001,
    gradientDescentMaxSteps = 1000, 
    epsilon = 0.001) {
    // Devraient être initialisés par le constructeur de la classe enfant.
    this.challengeId = challengeId;
    this.layersSizes = layers;
    this.numLayers = layers.length;
    this.numCopies = expectedOutputs.length;
    this.maxLayerSize = Math.max(...this.layersSizes);
    this.parameters = [];
    this.minError = minError;
    this.gradientDescentFactor = gradientDescentFactor;
    this.gradientDescentMaxSteps = gradientDescentMaxSteps;
    this.epsilon = epsilon;
    this.isActivated = false;
    this.isSolved = false;


    // Les copies du réseau avec les différents inputs/outputs
    this.nwi = new NetworkWithInputs(layers, inputs, expectedOutputs);

    // Les paramètres
    for (let paramId = 0; paramId < this.nwi.numWeights; paramId++) {
      this.parameters[paramId] = new Parameter(challengeId);
    }


    // Ajoute le challenge à la liste de tous les challenges
    // De plus, les challenges forment une liste chaînée via l'attribut
    // `nextChallenge`.
    Challenge.all[challengeId] = this;
    if (Challenge.lastAdded != null) {
      Challenge.lastAdded.nextChallenge = this;
    }
    Challenge.lastAdded = this;
    this.nextChallenge = null;

    // Le canvas est la zone où la courbe et les points sont dessinés.
    this.canvas = document.createElement("CANVAS");
    this.canvas.style.border = "2px solid black";

    // Paramètres d'affichage
    if ((this.numLayers == 2) && (this.numCopies == 1)) {
      this.canvas.width = 650;
      this.canvas.height = 300;
    }
    else if ((this.numLayers == 4) && (this.numCopies == 2)) {
      this.canvas.width = 750;
      this.canvas.height = 400;
    }
    else if ((this.numLayers == 2) && (this.numCopies == 2)) {
      this.canvas.width = 650;
      this.canvas.height = 400;
    }
    else if ((this.numLayers == 2) && (this.numCopies == 3)) {
      this.canvas.width = 650;
      this.canvas.height = 700;
    }
    else if ((this.numLayers == 3) && (this.numCopies == 4)) {
      this.canvas.width = 750;
      this.canvas.height = 750;
    }
    else if ((this.numLayers == 3) && (this.numCopies == 2)) {
      this.canvas.width = 750;
      this.canvas.height = 400;
    }
    else {
      throw new Error("No display settings for this kind of network");
    }
    this.xmin = -1;  // On veut garder un offset horizontal équivalent à une unité à gauche et à droite du premier/dernier neurone.
    this.xmax = 18;  // L'intervalle horizontal est donc de 12 + 2 unités de offset plus 5 autres unités pour afficher l'objectif.
    this.ymin = -10;
    this.ymax = 10;
    this.nodeRadius = 20;
    this.defaultLineWidth = 3;

    this.computeNeuronsPositions();

    // calcul automatique du facteur adhoc
    // rappel : ça sert à ramener l'erreur dans une fourchette d'environ 0 à 1.
    this.adhocErrorFactor = 1.0;
    this.adhocErrorFactor = 1/this.error() * 0.75;

    // Bouton indiquant que le défi est réussi
    this.buttonSolved = document.createElement("button");
    this.buttonSolved.className = "defiReussi";
    this.buttonSolved.innerHTML = "Défi réussi !!!";
    this.buttonSolved.style.visibility = "hidden";

    // Bouton pour lancer la solution automatique
    this.buttonAutoSolve = document.createElement("button");
    this.buttonAutoSolve.className = "reset";
    this.buttonAutoSolve.innerHTML = "Solution <br> Automatique";
    this.buttonAutoSolve.style.visibility = "hidden";
    this.buttonAutoSolve.onclick = async function() {
      await Challenge.all[challengeId].autoSolve();
    }

    // Bouton pour débloquer les contrôles
    this.buttonUnlock = document.createElement("button");
    this.buttonUnlock.className = "reset";
    this.buttonUnlock.innerHTML = "Débloquer";
    this.buttonUnlock.style.visibility = "hidden";
    this.buttonUnlock.onclick = function() {
      let challenge = Challenge.all[challengeId];
      challenge.unlock();
      challenge.buttonUnlock.style.visibility = "hidden";
      challenge.doNotLock = true;
      setTimeout(() => {challenge.doNotLock = false;}, 5000);
    }
    this.doNotLock = false;

    // Construction du thermomètre
    this.thermometer = document.createElement("CANVAS");
    this.thermometer.className = "thermometre";
    this.thermometer.style.border = "0px;";
    this.thermometer.style.visibility = 'hidden';
    this.thermometer.width = 100;
    this.thermometer.height = 300;

    this.grosseTriche = 1;

    this.addToDocument(); // ligne obligatoire
    this.update(); // ligne obligatoire
  }



  /**
   * Ajoute le défi dans le document HTML
   */
  addToDocument(introduction = "") {
    let div, table, col, line, button;

    div = document.getElementById(this.challengeId);
    this.intro = document.createElement("div");
    this.intro.innerHTML = introduction;
    this.intro.style.visibility = "hidden";
    div.appendChild(this.intro);
    table = document.createElement("table");
    line = document.createElement("tr");
    col = document.createElement("td");
    col.appendChild(this.canvas);
    line.appendChild(col)
    col = document.createElement("td");
    col.appendChild(this.thermometer);
    line.appendChild(col)
    table.appendChild(line);
    div.appendChild(table);

    table = document.createElement("table");
    for (let i=0; i<this.parameters.length; i++) {
      var p = this.parameters[i];
      line = document.createElement("tr");

      // 1ere colonne : le nom du paramètre
      col = document.createElement("td");
      col.innerHTML = String.fromCharCode("a".charCodeAt(0) + i)
      line.appendChild(col);

      // 2e colonne : le slider
      col = document.createElement("td");
      col.appendChild(p.slider);
      line.appendChild(col);

      //3e colonne : le slider
      col = document.createElement("td");
      col.appendChild(p.label);
      line.appendChild(col);

      line.appendChild(col);
      table.appendChild(line);
    }
    div.appendChild(table);

    let div2 = document.createElement("div");
    div2.style.display = "flex";
    div2.style.gap = "10px";
    div2.style.alignItems = "center";
    div2.appendChild(this.buttonSolved);
    div2.appendChild(this.buttonAutoSolve);
    div2.appendChild(this.buttonUnlock);
    div.appendChild(div2);

  }


  computeNeuronsPositions() {
    let copyDx;
    let copyDy;
    let dx;
    let dy;
    let x0;
    let y0;
    let targetExtraDx;
    if ((this.numCopies == 1) && (this.numLayers == 2) && (this.maxLayerSize == 1)) {
      copyDx = 0;
      copyDy = 2;
      dx = 10;
      dy = -6;
      x0 = 1;
      y0 = 0;
      targetExtraDx = -5;
    }
    else if ((this.numCopies == 2) && (this.numLayers == 4) && (this.maxLayerSize == 2)) {
      copyDx = 0;
      copyDy = -10;
      dx = 4;
      dy = -5;
      x0 = 1;
      y0 = 7;
      targetExtraDx = -2;
    }
    else if ((this.numCopies == 2) && (this.numLayers == 2) && (this.maxLayerSize == 2)) {
      copyDx = 0;
      copyDy = -10;
      dx = 10;
      dy = -4.3;
      x0 = 1;
      y0 = 8;
      targetExtraDx = -5;
    }
    else if ((this.numCopies == 2) && (this.numLayers == 2) && (this.maxLayerSize == 3)) {
      copyDx = 0;
      copyDy = -10;
      dx = 10;
      dy = -3.0;
      x0 = 1;
      y0 = 8;
      targetExtraDx = -5;
    }
    else if ((this.numCopies == 3) && (this.numLayers == 2) && (this.maxLayerSize == 2)) {
      copyDx = 0;
      copyDy = -7;
      dx = 10;
      dy = -2.5;
      x0 = 1;
      y0 = 8;
      targetExtraDx = -5;
    }
    else if ((this.numCopies == 4) && (this.numLayers == 3) && (this.maxLayerSize == 3)) {
      copyDx = 0;
      copyDy = -4.5;
      dx = 6;
      dy = -1.2;
      x0 = 1;
      y0 = 8;
      targetExtraDx = -2;
    }
    else if ((this.numCopies == 2) && (this.numLayers == 3) && (this.maxLayerSize == 2)) {
      copyDx = 0;
      copyDy = -8.5;
      dx = 6;
      dy = -3.2;
      x0 = 1;
      y0 = 8;
      targetExtraDx = -2;
    }
    else {
      throw new Error("Taille de défi non géré");
    }
    this.neuronsPositions = [];
    for (let copyId=0; copyId<this.numCopies; copyId++) {
      let x = x0;
      let y = y0;
      this.neuronsPositions[copyId] = [];
      for (let i=0; i<this.numLayers; i++) {
        this.neuronsPositions[copyId][i] = [];
        for (let j=0; j<this.layersSizes[i]; j++) {
          this.neuronsPositions[copyId][i][j] = [x, y];
          //console.log("position[" + copyId + "][" + i + "][" + j + "] = [" + x + ", " + y + "]");
          y += dy;
        }
        x += dx;
        y = y0;
      }
      // on ajoute un faux layer pour les `expected outputs`
      let i = this.numLayers;
      this.neuronsPositions[copyId][i] = [];
      x += targetExtraDx
      y = y0;
      for (let j=0; j<this.layersSizes[i-1]; j++) {
        this.neuronsPositions[copyId][i][j] = [x, y];
          //console.log("position[" + copyId + "][" + i + "][" + j + "] = [" + x + ", " + y + "]");
        y += dy;
      }
      x0 += copyDx;
      y0 += copyDy;
    }
  }

  /**
   * converti une coordonnée x pour qu'elle corresponde au système de coordonnées du canvas
   */
  convertX(x) {
    return (x - this.xmin) * this.canvas.width / (this.xmax - this.xmin);
  }

  /**
   * converti une coordonnée y pour qu'elle corresponde au système de coordonnées du canvas
   */
  convertY(y) {
    return this.canvas.height *(1.0 -  ((y - this.ymin) / (this.ymax - this.ymin)));
  }

  componentToHex(c)
  {
    let hex = c.toString(16);
    return hex.length == 1 ? "0" + hex : hex;
  }


  rgbToHex(r, g, b)
  {
    return "#" + this.componentToHex(r) + this.componentToHex(g) + this.componentToHex(b);

  }

  // Value est une valeur de gris entre 0 et 100, pour laquelle on recevra le code hex
  getGray(value)
  {
    let v = Math.trunc(255 * value / 100);
    return this.rgbToHex(v, v, v);
  }

  /**
   * Effectue la mise à jour du défi.
   * Entre autre, cette fonction est appelée à chaque que les paramètres sont modifiés.
   */
  update() {
    if (this.isActivated) {
      // mise à jour des labels des paramètres
      for (let i in this.parameters) {
        let p = this.parameters[i];
        p.label.innerHTML = p.slider.value;
        this.nwi.setWeight(i, parseFloat(p.slider.value));
      }
      this.nwi.propagate();
      this.drawSelf();
      this.drawThermometer();
      this.validate();
    }
    else {
      let ctx = this.canvas.getContext("2d");
      ctx.fillStyle = 'lightgray';
      ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      ctx.fill();
    }
  }

  /**
   * Dessine le réseau de neurones avec les poids par défaut dans le canvas.
   * 
   * Old, c'est devenu un fonction virtuelle qui doit absolument être définie par
   * les classes enfants.
   */
  drawSelf() 
  {
    //throw new Error('Cette fonction doit être redéfinie dans la classe enfant');
    //
    let ctx = this.canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0,0, this.canvas.width, this.canvas.height);

    // draw links
    for (let copyId=0; copyId<this.numCopies; copyId++) {
      for (let paramId=0; paramId<this.parameters.length; paramId++) {
        let indices = this.nwi.getWeightIndices(paramId);
        let i = indices[0];
        let j = indices[1];
        let k = indices[2];
        let pos0 = this.neuronsPositions[copyId][i][j];
        let pos1 = this.neuronsPositions[copyId][i+1][k];
        let x0 = this.convertX(pos0[0]);
        let y0 = this.convertY(pos0[1]);
        let x1 = this.convertX(pos1[0]);
        let y1 = this.convertY(pos1[1]);
        this.drawLinkBetweenNeurons(ctx, paramId, x0, y0, x1, y1);
      }
    }


    // draw neurons
    for (let copyId=0; copyId<this.numCopies; copyId++) {
      for (let i=0; i<this.numLayers; i++) {
        for (let j=0; j<this.layersSizes[i]; j++) {
          let pos = this.neuronsPositions[copyId][i][j];
          let x = pos[0];
          let y = pos[1];
          let activation = this.nwi.copies[copyId][i][j]
          this.drawNode(ctx, x, y, activation);
        }
      }
      for (let j=0; j<this.nwi.expectedOutputs[copyId].length; j++) {
        let pos = this.neuronsPositions[copyId][this.numLayers][j];
        let x = pos[0];
        let y = pos[1];
        let activation = this.nwi.expectedOutputs[copyId][j];
        this.drawNode(ctx, x, y, activation);
      }
    }

    // Finalement, on dessine la zone de sortie attendue
    ctx.lineWidth = this.defaultLineWidth;
    ctx.fillStyle = this.getGray(75);
    ctx.lineWidth = 2; 
    let x0 = this.neuronsPositions[0][this.numLayers-1][0][0];
    let x1 = this.neuronsPositions[0][this.numLayers][0][0];
    let x = (x0 + x1) / 2;
    let y0 = this.canvas.height / 10;
    let y1 = this.canvas.height - y0;
    ctx.moveTo(this.convertX(x), y0);
    ctx.lineTo(this.convertX(x), y1);
    ctx.stroke();

    ctx.fillStyle = "blue";
    ctx.font = "30px Arial";
    //ctx.fillText("Objectif", (this.convertX(x0) + this.canvas.width)/2, y0 - 30);
    //ctx.fillStyle = "black";
    //ctx.font = "20px";
    //ctx.fillText("a", this.convertX(6), this.convertY(0.6));
  }



  /**
   * Calcule l'erreur. En général, il s'agit de l'erreur quadratique moyenne,
   * mais cela dépent du défi.
   *
   * C'est pourquoi cette fonction doit obligatoirement être redéfinie dans la
   * classe enfant.
   */
  error() {
    return this.nwi.mse() * this.adhocErrorFactor;
    //return this.nwi.mse() * this.adhocErrorFactor * this.grosseTriche;
  }


  /**
   * Teste si le défi est complété. Si c'est le cas, on affiche qu'il est
   * complété et on débloque le défi suivant.
   */
  validate() {
    let error = this.error();
    if (error < this.minError) {
      this.buttonSolved.style.visibility = "visible";
      if (!this.doNotLock) {
        this.lock();
        this.buttonUnlock.style.visibility = "visible";
      }
    }
    else {
      this.buttonSolved.style.visibility = "hidden";
    }
    if (error < this.minError && !this.isSolved) {
      this.isSolved = true;
      if (this.nextChallenge != null) {
        this.nextChallenge.activate();
      }
    } 
    else {
      //console.log("echec : " + error);
    }
  }

  /**
   * Désactive les contrôles du défi.
   */
  lock() {
    for (var i in this.parameters) {
      this.parameters[i].slider.disabled = true;
    }
  }

  /**
   * Active les contrôles du défi.
   */
  unlock() {
    for (var i in this.parameters) {
      this.parameters[i].slider.disabled = false;
    }
    this.isSolved = false;
  }

  /**
   * Active le défi. Le joueur peut maintenant tenter de le résoudre.
   */
  activate() {
    this.intro.style.visibility = "visible";
    this.isActivated = true;
    this.unlock();
    let status = sessionStorage.getItem('status');
    if (status === null) {
      sessionStorage.setItem('status', '0');
    }
    status = parseInt(sessionStorage.getItem('status'));
    if (status >= 2) {
      this.buttonAutoSolve.style.visibility = "visible";
    }
    this.update();
  }


  /**
   * Retourne la valeur du paramètre spécifié
   */
  parameterValue(name) {
    return parseFloat(this.parameters[name].slider.value);
  }

  drawLinkBetweenNeurons(ctx, paramId, x0, y0, x1, y1) {
    let paramName = String.fromCharCode('a'.charCodeAt(0) + paramId)
    let w = this.parameterValue(paramId);	
    if (w < 0) {
      ctx.setLineDash([5, 5]);
    }
    ctx.lineWidth = Math.max(1, Math.abs(w)*10);
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
    ctx.setLineDash([]);

    if (this.nwi.numWeights <= 4) {
      let x = x0 + (x1 - x0) / 3;
      let y = y0 + (y1 - y0) / 3;
      ctx.font = "15px Arial";
      ctx.fillStyle = "black";
      ctx.fillText(paramName, x, y - 10);
    }
  }

  drawNode(ctx, x, y, activation) {
    ctx.lineWidth = this.defaultLineWidth;
    let X = this.convertX(x);
    let Y = this.convertY(y);

    ctx.beginPath();
    if (activation >= 0) {
      ctx.fillStyle = this.getGray(activation);
      ctx.arc(X, Y, this.nodeRadius, 0, 2*Math.PI, false);
      ctx.fill();
      ctx.style = "black";
      ctx.stroke();
    } else {
      ctx.fillStyle = this.getGray(-activation);
      ctx.arc(X, Y, this.nodeRadius, 0, 2*Math.PI, false);
      ctx.fill();
      ctx.stroke();
      ctx.arc(X, Y, this.nodeRadius, 0, 2*Math.PI, false);
      ctx.fillStyle = createHatchPattern(ctx);
      ctx.fill();
    }

  }

  drawThermometer() {
    let ctx = this.thermometer.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0,0, this.canvas.width, this.canvas.height);


    var topLeft = {x : 40, y : 20};
    var width = 20;
    var height = 260;

    var error = Math.max(0, Math.min(1.0, this.error()) - this.minError);;
    let blue = Math.max(0, Math.min(255, 255 - Math.trunc(255 * Math.pow(error, 2))));
    let red = Math.max(0, Math.min(255, 150+Math.trunc(255 * Math.pow(error, 2))));
    let mercury = Math.min(height, Math.trunc(Math.pow(error, 0.5) * height)); // hauteur du mercure dans le thermomètre

    // Dessiner le mercure
    ctx.beginPath();
    ctx.fillStyle = this.rgbToHex(red, 0, blue);
    ctx.rect(topLeft.x, topLeft.y + (height - mercury), width, mercury)
    ctx.fill();

    // Dessiner le contour
    ctx.beginPath();
    ctx.strokeStyle = "black";
    ctx.lineWidth = 3;
    ctx.rect(topLeft.x, topLeft.y, width, height);
    ctx.stroke();
  }

  gradientDescentOneIteration() {

    // On parcourt les paramètres en ordre inverse pour genre simuler
    // `back-propagation`.
    //let params = Object.keys(this.parameters);
    //params.sort().reverse();

    let errorBefore = this.error()
    for (let i=this.nwi.numWeights-1; i >= 0; i--) {
      let p = this.parameters[i];
      let w = this.nwi.getWeight(i);
      let e0 = this.nwi.mse();
      this.nwi.setWeight(i, w + this.epsilon);
      this.nwi.propagate();
      let e1 = this.nwi.mse();
      let gradientStep = this.gradientDescentFactor * (e1-e0);
      p.slider.value = parseFloat(p.slider.value) - gradientStep;
    }

    let errorAfter = this.error()
    if (errorAfter >= errorBefore) {
      //console.log('diminution du step')
      //this.gradientStep = this.gradientStep / 1.1;
      //this.grosseTriche = this.grosseTriche / 2;
      //throw new Error("L'erreur augmente!");
    }
    else {
      //this.gradientStep = this.gradientStep / 1.01;
    }
    //console.log("Error delta = " + (errorAfter - errorBefore));

  }

  async gradientDescent() {
    let refreshRate = 1;
    let timeout = 10;
    for (let i=0; i<this.gradientDescentMaxSteps; i++) {
      this.gradientDescentOneIteration();
      if (i % refreshRate == 0) {
        this.update();
        if (this.isSolved) {
          return;
        }
        if (timeout > 1) {
          await sleep(timeout);
        }
      }
    }
  }

  async autoSolve() {
    if (this.isSolved) {
      this.isSolved = false;
    }
    await this.gradientDescent();
  }

} // fin de classe Defi



// Fonctions pour cacher ou afficher les thermomètres
function hideThermometers() {
  for (let i in Challenge.all) {
    Challenge.all[i].thermometer.style.visibility = 'hidden';
  }
}
function thermometres() {
  for (let i in Challenge.all) {
    Challenge.all[i].thermometer.style.visibility = '';
  }
}







// Gestion des bouton pour activer le thermomètre
document.getElementById("cheatButton2").style.visibility = "hidden";
document.getElementById("cheatButton3").style.visibility = "hidden";
function cheatButton1Pressed() {
  document.getElementById("cheatButton2").style.visibility = "";
}
function cheatButton2Pressed() {
  document.getElementById("cheatButton3").style.visibility = "";
}
function cheatButton3Pressed() {
  document.getElementById("cheatButton1").classList.toggle("clignotant");
  document.getElementById("cheatButton2").classList.toggle("clignotant");
  document.getElementById("cheatButton3").classList.toggle("clignotant");
  document.getElementById("cheatText").style.visibility = "";
  document.getElementById("cheatText").classList.toggle("clignotant");
  sessionStorage.setItem("status", "1");
  thermometres();
}

// Gestion de la face surprise
const face = document.getElementById("face");
let countDown = 12;
let bliking = true;
face.addEventListener("click", () => {
  face.classList.toggle("surprised");
  // reset after 0.2 second
  setTimeout(() => {
    face.classList.toggle("surprised");
  }, 300);
  countDown--;
  if (countDown <= 10 && bliking) {
    document.getElementById("cheatText").classList.toggle("clignotant");
    bliking = false;
  }
  if (countDown <= 7) {
    document.getElementById('cheatText2').innerHTML = `Cliquez encore ${countDown} fois pour activer la solution automatique !`;
  }
  if (countDown <= 0) {
    document.getElementById('cheatText2').innerHTML = 'Mode <b>Solution Automatique</b> activé !<br> Retour à <button class=\"mission\" onclick=\"window.location.href=\'mission1.html\'\">Mission 1</button>';
    sessionStorage.setItem("status", "2");
    for (let i in Challenge.all) {
      if (Challenge.all[i].isActivated) {
        Challenge.all[i].activate();
      }
    }
  }
});


function updatePageAccordingToStatus() {
  // Gestion du status
  let status = sessionStorage.getItem('status');
  if (status === null) {
    sessionStorage.setItem('status', '0');
  }
  status = parseInt(sessionStorage.getItem('status'));
  if (status >= 1) {
    thermometres();
  }
}



// Construction des défis
var d1 = new Challenge("challenge1", 
  layers=[1,1], 
  inputs=[[100]], 
  expectedOutputs=[[65]], 
  minError=0.0001, 
  gradientDescentFactor = 0.004,
  gradientDescentMaxSteps = 1000,
  epsilon = 0.001
);

var d2 = new Challenge("challenge2", 
  layers=[2,2], 
  inputs=[[100, 0], [0, 100]],
  expectedOutputs=[[50, 50], [75, 15]],
  minError=0.02, 
  gradientDescentFactor = 0.002,
  gradientDescentMaxSteps = 1000,
  epsilon = 0.001
);

var d3 = new Challenge("challenge3", 
  layers=[3,2], 
  inputs=[[25, 66, 85], [88, 0, 95]],
  expectedOutputs=[[95, 25], [0,100]],
  minError=0.005, 
  gradientDescentFactor = 0.002,
  gradientDescentMaxSteps = 10000,
  epsilon = 0.002
);

var d4 = new Challenge("challenge4", 
  layers=[2,2,2], 
  inputs=[[100, 0], [0, 100]], 
  expectedOutputs=[[25, 75], [100, 100]], 
  minError=0.001, 
  gradientDescentFactor = 0.015,
  gradientDescentMaxSteps = 1000,
  epsilon = 0.001
);

var d5 = new Challenge("challenge5", 
  layers=[3,3,2], 
  inputs=[[100, 0, 100], [100, 100, 0], [0, 100, 100], [50, 100, 50]], 
  expectedOutputs=[[0,0], [75,25], [25,75], [50,33]], 
  minError=0.02, 
  gradientDescentFactor = 0.02,
  gradientDescentMaxSteps = 10000,
  epsilon = 0.001
);

// Activation du premier défi
Challenge.all['challenge1'].activate();
//Challenge.all['challenge2'].activate();
//Challenge.all['challenge3'].activate();
//Challenge.all['challenge4'].activate();
//



function activateAll() {
  for (let i in Challenge.all) {
    Challenge.all[i].activate();
  }
}


// debug
activateAll()

updatePageAccordingToStatus();
