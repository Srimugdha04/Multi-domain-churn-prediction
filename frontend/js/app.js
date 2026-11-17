/* ===================================================
   NEXA AI DASHBOARD — app.js
   All frontend logic: auth, data fetching, UI updates
   =================================================== */

// ---- App State ----
let user = JSON.parse(localStorage.getItem('user')) || null;
let latestPredictions = null;
let chartInstance = null;

// ---- DOM Ready ----
document.addEventListener('DOMContentLoaded', () => {
    // Restore session if user exists
    if (user) {
        let loginOverlay = document.getElementById('login-overlay');
        if (loginOverlay) loginOverlay.style.display = 'none';
        
        let userNameEl = document.getElementById('user-name');
        if (userNameEl) userNameEl.textContent = user.name;
        
        let userInitialsEl = document.getElementById('user-initials');
        if (userInitialsEl) userInitialsEl.textContent = user.name.split(' ').map(w => w[0]).join('').slice(0, 2);
        
        let topbarBankEl = document.getElementById('topbar-bank');
        if (topbarBankEl) topbarBankEl.textContent  = `Bank ID: ${user.bankId}`;
        
        // Only loadData if we are on the main dashboard to prevent error
        if (document.getElementById('ledger-body')) {
            loadData();
        }
    }

    // Allow pressing Enter in login fields
    let passInput = document.getElementById('pass');
    if (passInput) {
        passInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') login();
        });
    }
    let emailInput = document.getElementById('email');
    if (emailInput) {
        emailInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') login();
        });
    }
    // Allow pressing Enter to send chat
    let chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMsg();
        });
    }
});

// ===================================================
// AUTH
// ===================================================

async function login() {
    const email    = document.getElementById('email').value.trim();
    const pass     = document.getElementById('pass').value;
    const loginBtn = document.getElementById('login-btn');
    const loginErr = document.getElementById('login-error');

    if (!email || !pass) {
        showLoginError('Please enter both email and password.');
        return;
    }

    // Loading state
    loginBtn.innerHTML = '<span class="spinner"></span> Signing in...';
    loginBtn.disabled = true;
    loginErr.style.display = 'none';

    try {
        const res  = await fetch('/api/auth/login', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ email, password: pass })
        });
        const data = await res.json();

        if (res.ok && data.status === 'success') {
            user = data.data;
            localStorage.setItem('user', JSON.stringify(user));
            let loginOverlay = document.getElementById('login-overlay');
            if (loginOverlay) loginOverlay.style.display = 'none';

            let userNameEl = document.getElementById('user-name');
            if (userNameEl) userNameEl.textContent = user.name;

            let userInitialsEl = document.getElementById('user-initials');
            if (userInitialsEl) userInitialsEl.textContent = user.name.split(' ').map(w => w[0]).join('').slice(0, 2);

            let topbarBankEl = document.getElementById('topbar-bank');
            if (topbarBankEl) topbarBankEl.textContent = `Bank ID: ${user.bankId}`;
            
            if (document.getElementById('ledger-body')) {
                loadData();
            } else {
                // If not on the dashboard, redirect there
                window.location.href = '/';
            }
        } else {
            showLoginError('Invalid email or password. Please try again.');
        }
    } catch (err) {
        showLoginError('Cannot connect to server. Is Flask running?');
    } finally {
        loginBtn.innerHTML = 'Sign In';
        loginBtn.disabled  = false;
    }
}

function showLoginError(msg) {
    const el = document.getElementById('login-error');
    if (el) {
        el.textContent    = msg;
        el.style.display  = 'block';
    }
}


async function logout() {
    try {
        await fetch('/api/auth/logout', {
            method: 'POST'
        });
    } catch (err) {
        console.error('Logout request failed:', err);
    }

    localStorage.removeItem('user');
    user = null;
    latestPredictions = null;

    const overlay = document.getElementById('login-overlay');

    if (overlay) {
        overlay.style.display = 'flex';
    }

    const loginErr = document.getElementById('login-error');
    if (loginErr) loginErr.style.display = 'none';

    const emailInput = document.getElementById('email');
    if (emailInput) emailInput.value = '';

    const passInput = document.getElementById('pass');
    if (passInput) passInput.value = '';

    location.reload();
}

window.logout = logout;
function handleSessionExpired() {
    localStorage.removeItem('user');
    user = null;
    latestPredictions = null;

    alert('Your session has expired. Please sign in again.');
    window.location.reload();
}
// ===================================================
// NAVIGATION
// ===================================================

function switchTab(tab, el) {
    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
    el.classList.add('active');
    const titleMap = { dashboard: 'Bank Ledger', customers: 'Customer Directory' };
    document.getElementById('view-title').textContent = titleMap[tab] || 'Bank Ledger';
    loadData();
}

// ===================================================
// LOAD CUSTOMER DATA  →  GET /api/bank/<bank_id>
// ===================================================


async function loadData() {
    const tbody = document.getElementById('ledger-body');

    if (!user || !user.bankId) {
        handleSessionExpired();
        return;
    }

    tbody.innerHTML = `<tr><td colspan="4" class="empty-state">
        <i class="fas fa-circle-notch fa-spin"></i><br>Loading customers...
    </td></tr>`;

    try {
        const res = await fetch(`/api/bank/${user.bankId}`);

        if (res.status === 401) {
            handleSessionExpired();
            return;
        }

        if (!res.ok) {
            throw new Error(`Customer request failed (${res.status}).`);
        }

        const data = await res.json();

        if (!Array.isArray(data.customers)) {
            throw new Error('Invalid customer data received from server.');
        }

        document.getElementById('stat-total').textContent = data.customers.length;
        document.getElementById('stat-bank').textContent = data.bank_name || 'Unknown';

        if (data.customers.length === 0) {
            tbody.innerHTML = `<tr><td colspan="4" class="empty-state">
                <i class="fas fa-users-slash"></i><p>No customers found.</p>
            </td></tr>`;
            return;
        }

        tbody.innerHTML = data.customers.map(c => `
            <tr>
                <td>
                    <span class="customer-name">${c.name}</span>
                    <span class="customer-id">${c.customerId}</span>
                </td>
                <td>${c.tenure} Mo</td>
                <td>$${Number(c.balance).toLocaleString('en-US', {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2
                })}</td>
                <td id="risk-${c.customerId}">
                    <span class="risk-pending">
                        <i class="fas fa-clock"></i> Pending
                    </span>
                </td>
            </tr>
        `).join('');

    } catch (err) {
        console.error('[loadData ERROR]', err);

        tbody.innerHTML = `<tr><td colspan="4" style="text-align:center;color:#f64e60;padding:30px;">
            <i class="fas fa-exclamation-triangle"></i> Failed to load data.<br>
            <small style="color:#a2a3b7;font-size:0.75rem;"></small>
        </td></tr>`;

        tbody.querySelector('small').textContent = err.message;
    }
}

// ===================================================
// PREDICT ALL  →  GET /api/predict/all/<bank_id>
// ===================================================


async function predictAll() {
    const btn = document.getElementById('predict-btn');

    if (!user || !user.bankId) {
        handleSessionExpired();
        return;
    }

    btn.classList.add('loading');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Running Model...';

    try {
        const res = await fetch(`/api/predict/all/${user.bankId}`);

        if (res.status === 401) {
            handleSessionExpired();
            return;
        }

        if (!res.ok) {
            throw new Error(`Prediction request failed (${res.status}).`);
        }

        const predictions = await res.json();

        if (!Array.isArray(predictions)) {
            throw new Error('Invalid prediction data received from server.');
        }

        latestPredictions = predictions;

        let highRisk = 0;

        latestPredictions.forEach(r => {
            const cell = document.getElementById(`risk-${r.customerId}`);
            if (!cell) return;

            const probability = Number(r.prob);

            if (!Number.isFinite(probability) || probability < 0 || probability > 100) {
                cell.textContent = 'Invalid prediction';
                return;
            }

            const pct = probability.toFixed(2);
            const risk = r.risk_level;

            const riskStyles = {
                High: { className: 'risk-high', icon: 'fa-exclamation-circle' },
                Medium: { className: 'risk-medium', icon: 'fa-exclamation-triangle' },
                Low: { className: 'risk-safe', icon: 'fa-check-circle' }
            };

            const style = riskStyles[risk];

            if (!style) {
                cell.textContent = 'Unknown Risk';
                return;
            }

            if (risk === 'High') highRisk++;

            cell.innerHTML = `
                <span class="${style.className}">
                    <i class="fas ${style.icon}"></i>
                    ${pct}% ${risk} Risk
                </span>`;
        });

        document.getElementById('stat-highrisk').textContent = highRisk;

    } catch (err) {
        console.error('[predictAll ERROR]', err);
        alert(`Prediction failed: ${err.message}`);
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-brain"></i> PREDICT ALL CUSTOMER CHURN';
    }
}

// ===================================================
// GRAPHICAL REPRESENTATION
// ===================================================



async function showGraph() {
    if (!latestPredictions) {
        alert('Please run "PREDICT ALL CUSTOMER CHURN" first to generate data.');
        return;
    }

    document.getElementById('graph-modal').style.display = 'flex';

    // Sort customers by predicted churn probability
    const sortedPredictions = [...latestPredictions].sort(
        (a, b) => Number(a.prob) - Number(b.prob)
    );

    const labels = sortedPredictions.map(r => r.customerId);
    const data = sortedPredictions.map(r => Number(r.prob));

    const ctx = document.getElementById('churnChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Churn Probability',
                data: data,
                borderColor: '#1bc5bd',
                backgroundColor: 'rgba(27, 197, 189, 0.08)',
                borderWidth: 2,
                pointRadius: 0,
                pointHoverRadius: 5,
                pointHoverBackgroundColor: '#1bc5bd',
                fill: true,
                tension: 0.15
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        title: function(items) {
                            return 'Customer ID: ' + items[0].label;
                        },
                        label: function(context) {
                            return 'Churn Probability: ' +
                                context.parsed.y.toFixed(2) + '%';
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Customers (sorted by churn probability)'
                    },
                    ticks: {
                        display: false
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Churn Probability (%)'
                    }
                }
            }
        }
    });
}


function closeGraph() {
    document.getElementById('graph-modal').style.display = 'none';
}

// ===================================================
// AI CHAT  →  POST /api/chat
// ===================================================

function toggleBot() {
    const w = document.getElementById('bot-window');
    const isVisible = w.style.display === 'flex';
    w.style.display = isVisible ? 'none' : 'flex';
    if (!isVisible) {
        // Welcome message on first open
        const logs = document.getElementById('chat-logs');
        if (logs.children.length === 0) {
            appendBubble('ai', `Hello! I'm Nexa AI, your bank analytics advisor for ${user ? user.bankId : 'your bank'}. How can I help you today?`);
        }
        setTimeout(() => document.getElementById('chat-input').focus(), 100);
    }
}


async function sendMsg() {
    const input = document.getElementById('chat-input');
    const val = input.value.trim();

    if (!val) return;

    if (!user || !user.bankId) {
        appendBubble('ai', 'Please log in to access Nexa AI.');
        return;
    }

    appendBubble('user', val);
    input.value = '';

    // Safe animated typing indicator
    const typingId = appendBubble('ai', 'Thinking...', true);

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: val,
                bankId: user.bankId
            })
        });
        if (res.status === 401) {
    handleSessionExpired();
    return;
    }

        let data = {};

        try {
            data = await res.json();
        } catch {
            data = {};
        }

        const typingEl = document.getElementById(typingId);

        if (typingEl) {
            if (!res.ok) {
                typingEl.textContent =
                    typeof data.reply === 'string'
                        ? data.reply
                        : `Request failed (${res.status}). Please try again.`;
            } else {
                typingEl.textContent =
                    typeof data.reply === 'string' && data.reply.trim()
                        ? data.reply
                        : 'I could not generate a response. Please try again.';
            }
        }

    } catch (err) {
        const typingEl = document.getElementById(typingId);

        if (typingEl) {
            typingEl.textContent =
                'Connection error. Please try again.';
        }
    }

    const logs = document.getElementById('chat-logs');
    logs.scrollTop = logs.scrollHeight;
}


function appendBubble(type, text, isTyping = false) {
    const id = 'bubble-' + Date.now() + Math.random().toString(36).slice(2);
    const logs = document.getElementById('chat-logs');

    const div = document.createElement('div');
    div.className = `chat-bubble ${type}`;
    div.id = id;

    if (isTyping) {
        const icon = document.createElement('i');
        icon.className = 'fas fa-circle-notch fa-spin';
        icon.setAttribute('aria-hidden', 'true');

        div.appendChild(icon);
        div.appendChild(document.createTextNode(' ' + text));
    } else {
        div.textContent = String(text ?? '');
    }

    logs.appendChild(div);
    logs.scrollTop = logs.scrollHeight;

    return id;
}

// ==========================================
// CREATE CUSTOMER + IMMEDIATE CHURN PREDICTION
// ==========================================

function openCreateCustomer() {
    const modal = document.getElementById('create-customer-modal');
    const form = document.getElementById('create-customer-form');
    const error = document.getElementById('create-customer-error');

    form.reset();
    error.textContent = '';
    error.style.display = 'none';
    modal.style.display = 'flex';
}

function closeCreateCustomer() {
    document.getElementById('create-customer-modal').style.display = 'none';
}

async function submitNewCustomer(event) {
    event.preventDefault();

    if (!user || !user.bankId) {
        handleSessionExpired();
        return;
    }

    const form = document.getElementById('create-customer-form');
    const error = document.getElementById('create-customer-error');
    const button = document.getElementById('save-customer-btn');

    const payload = Object.fromEntries(new FormData(form).entries());

    const numericFields = [
        'CreditScore', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'Satisfaction Score', 'Point Earned'
    ];

    numericFields.forEach(field => {
        payload[field] = Number(payload[field]);
    });

    button.disabled = true;
    button.textContent = 'Saving & Assessing...';
    error.textContent = '';
    error.style.display = 'none';

    try {
        const response = await fetch(`/api/customers/${user.bankId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (response.status === 401) {
            handleSessionExpired();
            return;
        }

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || 'Unable to create customer.');
        }

        closeCreateCustomer();

        // Refresh the customer ledger
        await loadData();

        // Invalidate the old prediction list.
        // Predict All will include the newly created customer.
        latestPredictions = null;

        const prediction = data.prediction;

        alert(
            `Customer created successfully!\n\n` +
            `Customer ID: ${data.customer.customerId}\n` +
            `Name: ${data.customer.name}\n` +
            `Churn Probability: ${prediction.churn_probability_percent}%\n` +
            `Prediction: ${prediction.prediction}\n` +
            `Risk Level: ${prediction.risk_level}`
        );

    } catch (err) {
        error.textContent = err.message;
        error.style.display = 'block';

    } finally {
        button.disabled = false;
        button.innerHTML = '<i class="fas fa-brain"></i> Save & Assess Risk';
    }
}
