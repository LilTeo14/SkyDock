// Global tracking variables
let ws = null;
let reconnectInterval = 2000;
let isConnected = false;

// Calibration state
let markerSize = 5.0; // cm
let focalLengthFactor = 0.8;
let zeroX = 0.0;
let zeroY = 0.0;
let zeroZ = 0.0;
let isCalibrated = false;

// Telemetry state (raw and filtered)
let rawX = 0.0, rawY = 0.0, rawZ = 0.5;
let targetX = 0.0, targetY = 0.0, targetZ = -0.5;
let currentX = 0.0, currentY = 0.0, currentZ = -0.5;

// Rotation matrices and quaternions
const targetQuaternion = new THREE.Quaternion();
const currentQuaternion = new THREE.Quaternion();

// Interpolation smoothing factor
const lerpFactor = 0.15;

// DOM Elements
const connectionDot = document.getElementById('connection-dot');
const connectionText = document.getElementById('connection-text');
const trackingBadge = document.getElementById('tracking-badge');
const webcamStream = document.getElementById('webcam-stream');

const markerSizeInput = document.getElementById('marker-size-input');
const focalSlider = document.getElementById('focal-length-slider');
const focalVal = document.getElementById('focal-val');
const cameraSelect = document.getElementById('camera-select');
const btnZero = document.getElementById('btn-zero-calibrate');
const btnReset = document.getElementById('btn-reset-calibration');

// Video settings DOM elements
const autoContrastCheckbox = document.getElementById('auto-contrast-checkbox');
const brightnessSlider = document.getElementById('brightness-slider');
const brightnessVal = document.getElementById('brightness-val');
const contrastSlider = document.getElementById('contrast-slider');
const contrastVal = document.getElementById('contrast-val');
const showProcessedCheckbox = document.getElementById('show-processed-checkbox');

// Tab Elements
const tabButtons = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

// New Sliders and Labels
const claheClipSlider = document.getElementById('clahe-clip-slider');
const claheClipVal = document.getElementById('clahe-clip-val');
const claheGridSlider = document.getElementById('clahe-grid-slider');
const claheGridVal = document.getElementById('clahe-grid-val');
const noiseReductionSelect = document.getElementById('noise-reduction-select');

const polyApproxSlider = document.getElementById('poly-approx-slider');
const polyApproxVal = document.getElementById('poly-approx-val');
const minPerimeterSlider = document.getElementById('min-perimeter-slider');
const minPerimeterVal = document.getElementById('min-perimeter-val');
const maxBorderErrSlider = document.getElementById('max-border-err-slider');
const maxBorderErrVal = document.getElementById('max-border-err-val');
const errCorrectionSlider = document.getElementById('err-correction-slider');
const errCorrectionVal = document.getElementById('err-correction-val');

// New Robustness UI Controls
const bilateralCheckbox = document.getElementById('bilateral-checkbox');
const morphologyCheckbox = document.getElementById('morphology-checkbox');
const morphologyKernelSlider = document.getElementById('morphology-kernel-slider');
const morphologyKernelVal = document.getElementById('morphology-kernel-val');
const roiCheckbox = document.getElementById('roi-checkbox');
const kalmanCheckbox = document.getElementById('kalman-checkbox');
const kalmanQSlider = document.getElementById('kalman-q-slider');
const kalmanQVal = document.getElementById('kalman-q-val');
const kalmanRSlider = document.getElementById('kalman-r-slider');
const kalmanRVal = document.getElementById('kalman-r-val');
const trackingModeSelect = document.getElementById('tracking-mode-select');

// Telemetry fields
const telX = document.getElementById('telemetry-x');
const telY = document.getElementById('telemetry-y');
const telZ = document.getElementById('telemetry-z');
const barX = document.getElementById('bar-x');
const barY = document.getElementById('bar-y');
const barZ = document.getElementById('bar-z');

const rotPitch = document.getElementById('val-pitch');
const rotYaw = document.getElementById('val-yaw');
const rotRoll = document.getElementById('val-roll');

// Three.js variables
let scene, camera, renderer, controls;
let phoneMesh, cameraModel, zeroMarker;
let padBorder, guideBeam;
let gridHelper, wallGridHelper;

// Initialize Three.js Scene
function initThree() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Scene
    scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x070913, 0.015);

    // Camera
    camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 100);
    // Position camera at an angle looking down at the workspace
    camera.position.set(0, 0.35, 0.5);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2 + 0.1; // Don't go below ground level
    controls.minDistance = 0.1;
    controls.maxDistance = 5;
    controls.target.set(0, 0, -0.4);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
    dirLight.position.set(5, 10, 7);
    dirLight.castShadow = true;
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0x00f0ff, 1, 2);
    pointLight.position.set(0, 0.2, -0.3);
    scene.add(pointLight);

    // Grid Floor
    gridHelper = new THREE.GridHelper(2, 40, 0x38bdf8, 0x1e293b);
    gridHelper.position.y = -0.25;
    scene.add(gridHelper);

    // Camera model at origin
    createCameraModel();

    // Smartphone model
    createPhoneModel();

    // Zero calibration marker (hidden by default)
    createZeroMarker();

    // Window resize handler
    window.addEventListener('resize', onWindowResize);

    // Start animation loop
    animate();
}

// Create a visual model representing the computer webcam
function createCameraModel() {
    const camGroup = new THREE.Group();

    // Lens cone
    const coneGeom = new THREE.ConeGeometry(0.015, 0.03, 16);
    coneGeom.rotateX(Math.PI / 2); // Point along -Z
    const coneMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.8 });
    const cone = new THREE.Mesh(coneGeom, coneMat);
    camGroup.add(cone);

    // Body box
    const boxGeom = new THREE.BoxGeometry(0.04, 0.02, 0.02);
    const boxMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.6 });
    const box = new THREE.Mesh(boxGeom, boxMat);
    box.position.z = 0.015;
    camGroup.add(box);

    // Glowing active indicator
    const glowGeom = new THREE.SphereGeometry(0.003, 8, 8);
    const glowMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff });
    const glow = new THREE.Mesh(glowGeom, glowMat);
    glow.position.set(0.012, 0.005, 0.005);
    camGroup.add(glow);

    camGroup.position.set(0, 0, 0);
    scene.add(camGroup);
}

// Create the landing pad model with the ArUco marker texture
function createPhoneModel() {
    const boardSize = 0.1333; // 5cm marker represents 13.33cm total board size
    const boardThickness = 0.003; // 3mm thin plate

    const loader = new THREE.TextureLoader();
    // Load ArUco marker image generated by backend
    const markerTexture = loader.load('/marker_0.png');
    markerTexture.generateMipmaps = true;
    markerTexture.minFilter = THREE.LinearMipmapLinearFilter;

    // Materials for each box face
    // BoxGeometry material index order: Right, Left, Top, Bottom, Front, Back
    const materials = [
        new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.2 }), // Right
        new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.2 }), // Left
        new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.2 }), // Top
        new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.8, roughness: 0.2 }), // Bottom
        new THREE.MeshBasicMaterial({ map: markerTexture }),                                // Front (Screen)
        new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.9, roughness: 0.1 })  // Back
    ];

    const geometry = new THREE.BoxGeometry(boardSize, boardSize, boardThickness);
    phoneMesh = new THREE.Mesh(geometry, materials);
    phoneMesh.castShadow = true;
    phoneMesh.receiveShadow = true;
    
    // Initial position
    phoneMesh.position.set(0, 0, -0.4);
    scene.add(phoneMesh);

    // Add local axis helper to the phone to see Pitch/Yaw/Roll visually
    const axesHelper = new THREE.AxesHelper(0.05);
    phoneMesh.add(axesHelper);

    // Create outline border highlighting the landing pad corners
    const edges = new THREE.EdgesGeometry(geometry);
    padBorder = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 }));
    phoneMesh.add(padBorder);

    // Create guide beam (line tracking the landing pad center from camera/origin)
    const beamGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 0, -0.4)
    ]);
    guideBeam = new THREE.Line(beamGeometry, new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2, transparent: true, opacity: 0.8 }));
    scene.add(guideBeam);
}

// Create the zero-position marker (holographic wireframe sphere)
function createZeroMarker() {
    const geometry = new THREE.SphereGeometry(0.02, 16, 16);
    const material = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        wireframe: true,
        transparent: true,
        opacity: 0.0
    });
    zeroMarker = new THREE.Mesh(geometry, material);
    scene.add(zeroMarker);
    
    // Add green coordinate lines at zero point
    const axesHelper = new THREE.AxesHelper(0.04);
    zeroMarker.add(axesHelper);
}

function onWindowResize() {
    const container = document.getElementById('three-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
}

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    // Apply linear interpolation (LERP) for smooth transition
    if (isConnected) {
        currentX += (targetX - currentX) * lerpFactor;
        currentY += (targetY - currentY) * lerpFactor;
        currentZ += (targetZ - currentZ) * lerpFactor;
        
        phoneMesh.position.set(currentX, currentY, currentZ);
        
        // Spherical linear interpolation (SLERP) for rotation
        currentQuaternion.slerp(targetQuaternion, lerpFactor);
        phoneMesh.quaternion.copy(currentQuaternion);
        
        // Update guide beam position
        if (guideBeam) {
            const posAttr = guideBeam.geometry.attributes.position;
            posAttr.array[3] = currentX;
            posAttr.array[4] = currentY;
            posAttr.array[5] = currentZ;
            posAttr.needsUpdate = true;
        }

        // Set dynamic colors and opacity based on tracking state
        let targetColor = 0x10b981; // Green
        let beamOpacity = 0.8;
        let isSolid = false;

        if (trackingBadge.classList.contains('estimating')) {
            targetColor = 0xf59e0b; // Amber
            beamOpacity = 0.6;
            isSolid = true;
        } else if (trackingBadge.classList.contains('tracking')) {
            targetColor = 0x10b981; // Green
            beamOpacity = 0.8;
            isSolid = true;
        } else {
            targetColor = 0xef4444; // Red
            beamOpacity = 0.15;
            isSolid = false;
        }

        if (padBorder) {
            padBorder.material.color.setHex(targetColor);
        }
        if (guideBeam) {
            guideBeam.material.color.setHex(targetColor);
            guideBeam.material.opacity = beamOpacity;
        }

        if (isSolid) {
            phoneMesh.material.forEach(m => {
                m.transparent = false;
                m.opacity = 1.0;
            });
        } else {
            phoneMesh.material.forEach(m => {
                m.transparent = true;
                m.opacity = 0.3; // Faded out when signal is lost
            });
        }
    }

    controls.update();
    renderer.render(scene, camera);
}

// Connect to WebSocket Server
function connectWebSocket() {
    console.log("Connecting to WebSocket...");
    ws = new WebSocket("ws://localhost:8765");

    ws.onopen = function () {
        console.log("WebSocket connected.");
        isConnected = true;
        connectionDot.className = "status-dot connected";
        connectionText.innerText = "Conectado";
        
        // Send current calibration settings on load
        sendCalibration();
        sendVideoSettings();
    };

    ws.onmessage = function (event) {
        const data = JSON.parse(event.data);
        
        // Handle camera list message from backend
        if (data.type === "camera_list") {
            cameraSelect.innerHTML = "";
            if (data.cameras && data.cameras.length > 0) {
                data.cameras.forEach(idx => {
                    const opt = document.createElement('option');
                    opt.value = idx;
                    opt.innerText = `Cámara ${idx} ${idx === 0 ? '(Predeterminada)' : ''}`;
                    cameraSelect.appendChild(opt);
                });
            } else {
                const opt = document.createElement('option');
                opt.value = 0;
                opt.innerText = "Ninguna cámara activa";
                cameraSelect.appendChild(opt);
            }
            return;
        }
        
        // Update webcam frame
        if (data.frame) {
            webcamStream.src = "data:image/jpeg;base64," + data.frame;
        }

        // Update tracking status
        if (data.detected) {
            if (data.kalman_active) {
                trackingBadge.className = "card-badge estimating";
                trackingBadge.innerText = "Estimando (Kalman)";
            } else if (data.tracked_marker === 1) {
                trackingBadge.className = "card-badge tracking";
                trackingBadge.innerText = "Rastreado (ID 1)";
            } else if (data.tracked_marker === 0) {
                trackingBadge.className = "card-badge tracking";
                trackingBadge.innerText = "Rastreado (ID 0)";
            } else {
                trackingBadge.className = "card-badge tracking";
                trackingBadge.innerText = "Rastreado";
            }
            
            // Raw coordinates in meters
            rawX = data.x;
            rawY = data.y;
            rawZ = data.z;

            // Target positions in Three.js (OpenCV Y is inverted, OpenCV Z is -Z in Three.js)
            targetX = rawX;
            targetY = -rawY;
            targetZ = -rawZ;

            // Rotation matrix mapping
            const r = data.rotation_matrix;
            const m = new THREE.Matrix4();
            // Flip coordinates corresponding to Three.js axis space (T * R_cv * T)
            m.set(
                 r[0], -r[1], -r[2], 0,
                -r[3],  r[4],  r[5], 0,
                -r[6],  r[7],  r[8], 0,
                    0,     0,     0, 1
            );
            targetQuaternion.setFromRotationMatrix(m);

            // Compute relative calibrated coordinates (in cm)
            const calX_cm = (rawX - zeroX) * 100;
            const calY_cm = (-rawY - zeroY) * 100; // using tared Y in Three.js coordinate system
            const calZ_cm = (rawZ - zeroZ) * 100;

            // Update Telemetry display
            updateTelemetry(calX_cm, calY_cm, calZ_cm, rawZ * 100, data.pitch, data.yaw, data.roll);
        } else {
            trackingBadge.className = "card-badge lost";
            trackingBadge.innerText = "Sin Señal";
        }
    };

    ws.onclose = function () {
        console.log("WebSocket connection closed. Retrying...");
        cleanupConnection();
        setTimeout(connectWebSocket, reconnectInterval);
    };

    ws.onerror = function (error) {
        console.error("WebSocket error:", error);
        ws.close();
    };
}

function cleanupConnection() {
    isConnected = false;
    connectionDot.className = "status-dot disconnected";
    connectionText.innerText = "Desconectado";
    trackingBadge.className = "card-badge lost";
    trackingBadge.innerText = "Sin Señal";
    webcamStream.src = "";
}

// Update DOM Telemetry values & progress bars
function updateTelemetry(cx, cy, cz, absZ, pitchRad, yawRad, rollRad) {
    // 1. Digital Values
    telX.querySelector('.telemetry-val').innerHTML = `${cx.toFixed(1)} <span class="telemetry-unit">cm</span>`;
    telY.querySelector('.telemetry-val').innerHTML = `${cy.toFixed(1)} <span class="telemetry-unit">cm</span>`;
    
    // For Z, we display absolute distance and show the relative drift in parenthesis
    const relSign = cz >= 0 ? '+' : '';
    telZ.querySelector('.telemetry-val').innerHTML = `${absZ.toFixed(1)} <span class="telemetry-unit">cm (${relSign}${cz.toFixed(1)})</span>`;

    // Rotations (Pitch, Yaw, Roll)
    const pDeg = (pitchRad * 180 / Math.PI).toFixed(0);
    const yDeg = (yawRad * 180 / Math.PI).toFixed(0);
    const rDeg = (rollRad * 180 / Math.PI).toFixed(0);
    rotPitch.innerText = `${pDeg}°`;
    rotYaw.innerText = `${yDeg}°`;
    rotRoll.innerText = `${rDeg}°`;

    // 2. Bar animations
    // X Bar: Centered bar, maps [-30cm, 30cm] range to [0%, 100%]
    const xMaxRange = 30.0;
    const xPercent = Math.min(50, (Math.abs(cx) / xMaxRange) * 50);
    if (cx >= 0) {
        barX.style.left = "50%";
        barX.style.width = `${xPercent}%`;
    } else {
        barX.style.left = `${50 - xPercent}%`;
        barX.style.width = `${xPercent}%`;
    }

    // Y Bar: Centered bar, maps [-20cm, 20cm] range
    const yMaxRange = 20.0;
    const yPercent = Math.min(50, (Math.abs(cy) / yMaxRange) * 50);
    if (cy >= 0) {
        barY.style.left = "50%";
        barY.style.width = `${yPercent}%`;
    } else {
        barY.style.left = `${50 - yPercent}%`;
        barY.style.width = `${yPercent}%`;
    }

    // Z Bar: Left-filled bar, maps absolute Z distance from [15cm, 120cm]
    const zMin = 15.0;
    const zMax = 120.0;
    const zPercent = Math.max(0, Math.min(100, ((absZ - zMin) / (zMax - zMin)) * 100));
    barZ.style.width = `${zPercent}%`;
}

// Send updated calibration parameters to python server
function sendCalibration() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const payload = {
            type: "calibrate",
            marker_size: markerSize / 100.0, // convert cm to meters
            focal_length_factor: focalLengthFactor
        };
        ws.send(JSON.stringify(payload));
        console.log("Sent calibration:", payload);
    }
}

// Send updated video settings to python server
function sendVideoSettings() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const payload = {
            type: "settings",
            brightness: parseFloat(brightnessSlider.value),
            contrast: parseFloat(contrastSlider.value),
            auto_contrast: autoContrastCheckbox.checked,
            show_processed: showProcessedCheckbox.checked,
            clahe_clip_limit: parseFloat(claheClipSlider.value),
            clahe_grid_size: parseInt(claheGridSlider.value),
            noise_reduction: parseInt(noiseReductionSelect.value),
            min_marker_perimeter_rate: parseFloat(minPerimeterSlider.value),
            poly_approx_accuracy_rate: parseFloat(polyApproxSlider.value),
            max_erroneous_bits_border: parseFloat(maxBorderErrSlider.value),
            error_correction_rate: parseFloat(errCorrectionSlider.value),
            roi_tracking_enabled: roiCheckbox.checked,
            bilateral_filtering: bilateralCheckbox.checked,
            morphology_enabled: morphologyCheckbox.checked,
            morphology_kernel_size: parseInt(morphologyKernelSlider.value),
            kalman_enabled: kalmanCheckbox.checked,
            kalman_q: parseFloat(kalmanQSlider.value),
            kalman_r: parseFloat(kalmanRSlider.value),
            tracking_mode: trackingModeSelect.value
        };
        ws.send(JSON.stringify(payload));
        console.log("Sent video settings:", payload);
    }
}

// Send camera change request to python server
function sendCameraChange(index) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const payload = {
            type: "change_camera",
            index: index
        };
        ws.send(JSON.stringify(payload));
        console.log("Sent camera change request:", payload);
    }
}

// Setup Event Listeners
function setupEvents() {
    // Tabs switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const targetTab = document.getElementById(btn.getAttribute('data-tab'));
            if (targetTab) {
                targetTab.classList.add('active');
            }
        });
    });

    // Camera selection change
    cameraSelect.addEventListener('change', () => {
        const index = parseInt(cameraSelect.value);
        sendCameraChange(index);
    });

    // Auto-contrast checkbox
    autoContrastCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    // CLAHE Clip Limit slider
    claheClipSlider.addEventListener('input', () => {
        claheClipVal.innerText = parseFloat(claheClipSlider.value).toFixed(1);
        sendVideoSettings();
    });

    // CLAHE Grid Size slider
    claheGridSlider.addEventListener('input', () => {
        claheGridVal.innerText = claheGridSlider.value;
        sendVideoSettings();
    });

    // Brightness slider
    brightnessSlider.addEventListener('input', () => {
        brightnessVal.innerText = brightnessSlider.value;
        sendVideoSettings();
    });

    // Contrast slider
    contrastSlider.addEventListener('input', () => {
        contrastVal.innerText = parseFloat(contrastSlider.value).toFixed(1);
        sendVideoSettings();
    });

    // Show processed checkbox
    showProcessedCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    // Noise reduction select
    noiseReductionSelect.addEventListener('change', () => {
        sendVideoSettings();
    });

    // Polygon Approximation slider
    polyApproxSlider.addEventListener('input', () => {
        polyApproxVal.innerText = parseFloat(polyApproxSlider.value).toFixed(3);
        sendVideoSettings();
    });

    // Min Perimeter slider
    minPerimeterSlider.addEventListener('input', () => {
        minPerimeterVal.innerText = parseFloat(minPerimeterSlider.value).toFixed(3);
        sendVideoSettings();
    });

    // Max Border Error slider
    maxBorderErrSlider.addEventListener('input', () => {
        maxBorderErrVal.innerText = parseFloat(maxBorderErrSlider.value).toFixed(2);
        sendVideoSettings();
    });

    // Error Correction slider
    errCorrectionSlider.addEventListener('input', () => {
        errCorrectionVal.innerText = parseFloat(errCorrectionSlider.value).toFixed(2);
        sendVideoSettings();
    });

    // Robustness Event Listeners
    bilateralCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    morphologyCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    morphologyKernelSlider.addEventListener('input', () => {
        morphologyKernelVal.innerText = morphologyKernelSlider.value;
        sendVideoSettings();
    });

    roiCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    kalmanCheckbox.addEventListener('change', () => {
        sendVideoSettings();
    });

    kalmanQSlider.addEventListener('input', () => {
        kalmanQVal.innerText = parseFloat(kalmanQSlider.value).toFixed(3);
        sendVideoSettings();
    });

    kalmanRSlider.addEventListener('input', () => {
        kalmanRVal.innerText = parseFloat(kalmanRSlider.value).toFixed(2);
        sendVideoSettings();
    });

    trackingModeSelect.addEventListener('change', () => {
        sendVideoSettings();
    });
    // Marker size input change
    markerSizeInput.addEventListener('change', () => {
        let val = parseFloat(markerSizeInput.value);
        if (isNaN(val) || val < 1.0) val = 5.0;
        markerSize = val;
        markerSizeInput.value = val.toFixed(1);
        sendCalibration();
        
        // Resize threejs phone mesh dimensions
        if (phoneMesh) {
            // scale base dimensions: width = 7cm, height = 14cm
            // scale factor based on physical marker sizing relative to 5cm
            const scale = val / 5.0;
            phoneMesh.scale.set(scale, scale, scale);
        }
    });

    // Focal length slider change
    focalSlider.addEventListener('input', () => {
        const val = parseFloat(focalSlider.value);
        focalLengthFactor = val;
        focalVal.innerText = val.toFixed(2);
        sendCalibration();
    });

    // Zero button (Tare position)
    btnZero.addEventListener('click', () => {
        zeroX = rawX;
        zeroY = -rawY; // store tared Y in Three.js coordinates
        zeroZ = rawZ;
        isCalibrated = true;
        
        // Update zero marker in Three.js
        zeroMarker.position.set(zeroX, zeroY, -zeroZ);
        zeroMarker.material.opacity = 0.5; // Make the wireframe target visible!
        
        console.log(`System calibrated at Zero Center: X=${zeroX.toFixed(3)}, Y=${zeroY.toFixed(3)}, Z=${zeroZ.toFixed(3)}`);
    });

    // Reset button
    btnReset.addEventListener('click', () => {
        zeroX = 0.0;
        zeroY = 0.0;
        zeroZ = 0.0;
        isCalibrated = false;
        
        // Hide zero marker
        zeroMarker.material.opacity = 0.0;
        
        // Reset slider & inputs
        markerSize = 5.0;
        markerSizeInput.value = "5.0";
        focalLengthFactor = 0.8;
        focalSlider.value = "0.80";
        focalVal.innerText = "0.80";
        
        // Reset video settings
        autoContrastCheckbox.checked = true;
        brightnessSlider.value = "0";
        brightnessVal.innerText = "0";
        contrastSlider.value = "1.0";
        contrastVal.innerText = "1.0";
        showProcessedCheckbox.checked = false;
        noiseReductionSelect.value = "0";
        
        // Reset advanced sliders
        claheClipSlider.value = "3.0";
        claheClipVal.innerText = "3.0";
        claheGridSlider.value = "8";
        claheGridVal.innerText = "8";
        
        polyApproxSlider.value = "0.055";
        polyApproxVal.innerText = "0.055";
        minPerimeterSlider.value = "0.015";
        minPerimeterVal.innerText = "0.015";
        maxBorderErrSlider.value = "0.50";
        maxBorderErrVal.innerText = "0.50";
        errCorrectionSlider.value = "0.80";
        errCorrectionVal.innerText = "0.80";

        // Reset robustness settings
        bilateralCheckbox.checked = true;
        morphologyCheckbox.checked = true;
        morphologyKernelSlider.value = "3";
        morphologyKernelVal.innerText = "3";
        roiCheckbox.checked = false;
        kalmanCheckbox.checked = true;
        kalmanQSlider.value = "0.050";
        kalmanQVal.innerText = "0.050";
        kalmanRSlider.value = "0.15";
        kalmanRVal.innerText = "0.15";
        trackingModeSelect.value = "dual";
        
        // Reset camera selection
        cameraSelect.value = "0";
        
        sendCalibration();
        sendVideoSettings();
        sendCameraChange(0);
        
        // Switch to the first tab (Position) on reset
        tabButtons.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        tabButtons[0].classList.add('active');
        tabContents[0].classList.add('active');
        
        if (phoneMesh) {
            phoneMesh.scale.set(1, 1, 1);
        }
        
        console.log("Calibration, video settings, advanced parameters and camera reset to defaults.");
    });
}

// Entry Point
window.onload = () => {
    initThree();
    setupEvents();
    connectWebSocket();
};
