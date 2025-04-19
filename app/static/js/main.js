// Main JavaScript file for the dashboard

// Handle filter changes
document.addEventListener('DOMContentLoaded', function() {
    // Get filter elements
    const industryFilter = document.getElementById('industry');
    const timeFilter = document.getElementById('time-period');
    const categoryFilter = document.getElementById('skill-category');
    
    // Add event listeners
    if (industryFilter) {
        industryFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    if (timeFilter) {
        timeFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    if (categoryFilter) {
        categoryFilter.addEventListener('change', function() {
            updateCharts();
        });
    }
    
    // Function to update charts based on filters
    function updateCharts() {
        const industry = industryFilter ? industryFilter.value : 'all';
        const timePeriod = timeFilter ? timeFilter.value : 'all';
        const category = categoryFilter ? categoryFilter.value : 'all';
        
        // Update rising skills chart
        if (document.getElementById('rising-skills-chart')) {
            fetch(`/api/rising-skills?industry=${industry}&time=${timePeriod}&category=${category}`)
                .then(response => response.json())
                .then(data => {
                    Plotly.react('rising-skills-chart', [{
                        x: data.skills,
                        y: data.values,
                        type: 'bar',
                        marker: {
                            color: '#0055ff'
                        }
                    }], {
                        margin: { t: 10, r: 10, b: 50, l: 50 },
                        yaxis: {
                            title: 'Growth Rate',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {
                            color: 'rgba(255,255,255,0.8)'
                        }
                    });
                });
        }
        
        // Update declining skills chart
        if (document.getElementById('declining-skills-chart')) {
            fetch(`/api/declining-skills?industry=${industry}&time=${timePeriod}&category=${category}`)
                .then(response => response.json())
                .then(data => {
                    Plotly.react('declining-skills-chart', [{
                        x: data.skills,
                        y: data.values,
                        type: 'bar',
                        marker: {
                            color: '#ff5555'
                        }
                    }], {
                        margin: { t: 10, r: 10, b: 50, l: 50 },
                        yaxis: {
                            title: 'Decline Rate',
                            gridcolor: 'rgba(255,255,255,0.1)'
                        },
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        plot_bgcolor: 'rgba(0,0,0,0)',
                        font: {
                            color: 'rgba(255,255,255,0.8)'
                        }
                    });
                });
        }
        
        // Update skill dropdown
        if (document.getElementById('skill-detail-dropdown')) {
            fetch(`/api/skills?category=${category}`)
                .then(response => response.json())
                .then(data => {
                    const dropdown = document.getElementById('skill-detail-dropdown');
                    
                    // Clear existing options
                    dropdown.innerHTML = '<option value="">Select a skill...</option>';
                    
                    // Add new options
                    data.skills.forEach(skill => {
                        const option = document.createElement('option');
                        option.value = skill;
                        option.textContent = skill;
                        dropdown.appendChild(option);
                    });
                });
        }
    }
});

// Add subtle animation effects
document.addEventListener('DOMContentLoaded', function() {
    // Animate chart containers on hover
    const chartContainers = document.querySelectorAll('.chart-container');
    chartContainers.forEach(container => {
        container.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 8px 30px rgba(0, 0, 0, 0.3)';
        });
        
        container.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = 'none';
        });
    });
});

// Add this to the end of main.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle Generate Recommendations button click
    const generateBtn = document.getElementById('generate-recommendations-button');
    if (generateBtn) {
        generateBtn.addEventListener('click', function() {
            // Get selected skills and target job
            const skillsElement = document.querySelector('.current-skills-list');
            const skills = skillsElement ? Array.from(skillsElement.querySelectorAll('li')).map(item => item.textContent) : [];
            
            const targetJob = document.getElementById('target-job-dropdown')?.value;
            const timeHorizon = document.getElementById('time-horizon-dropdown')?.value;
            
            if (!targetJob) {
                alert('Please select a target job role');
                return;
            }
            
            // Show loading state
            const recommendationsContainer = document.getElementById('recommendations-container');
            if (recommendationsContainer) {
                recommendationsContainer.innerHTML = '<div class="loading-spinner"></div>';
            }
            
            const pathChart = document.getElementById('skill-path-chart');
            if (pathChart) {
                pathChart.innerHTML = '<div class="loading-spinner"></div>';
            }
            
            // Fetch recommendations (replace with actual API endpoint)
            fetch(`/api/recommendations?target=${targetJob}&time=${timeHorizon}${skills.map(s => `&skill=${s}`).join('')}`)
                .then(response => response.json())
                .then(data => {
                    // Update recommendations
                    updateRecommendations(data);
                    
                    // Update path chart
                    updateSkillPathChart(data);
                })
                .catch(error => {
                    console.error('Error fetching recommendations:', error);
                    if (recommendationsContainer) {
                        recommendationsContainer.innerHTML = '<p>Error loading recommendations. Please try again.</p>';
                    }
                });
        });
    }
    
    // Initialize skill network graph
    initializeSkillNetwork();
});

function updateRecommendations(data) {
    const container = document.getElementById('recommendations-container');
    if (!container) return;
    
    // If no data or no recommendations
    if (!data || !data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = '<p>No specific recommendations found. You may already have the required skills for this role.</p>';
        return;
    }
    
    // Create recommendations HTML
    let html = '';
    data.recommendations.forEach((rec, index) => {
        html += `
            <div class="recommendation-item">
                <h4>${index + 1}. ${rec.skill}</h4>
                <p>Trend: ${rec.trend}</p>
                <p>Resources: ${rec.resources}</p>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateSkillPathChart(data) {
    const chartElement = document.getElementById('skill-path-chart');
    if (!chartElement) return;
    
    // Create a simple visualization if Plotly is available
    if (window.Plotly) {
        const currentSkills = data.currentSkills || [];
        const recommendedSkills = data.recommendations ? data.recommendations.map(r => r.skill) : [];
        
        // Create Sankey data
        const nodes = [
            { name: "Current Skills" },
            ...currentSkills.map(s => ({ name: s })),
            { name: "Target Skills" },
            ...recommendedSkills.map(s => ({ name: s })),
            { name: data.targetJob || "Career Goal" }
        ];
        
        const links = [
            // Link from "Current Skills" to each current skill
            ...currentSkills.map((_, i) => ({
                source: 0,
                target: i + 1,
                value: 1
            })),
            // Link from "Target Skills" to each recommended skill
            ...recommendedSkills.map((_, i) => ({
                source: currentSkills.length + 1,
                target: currentSkills.length + 2 + i,
                value: 1
            })),
            // Link from each recommended skill to the career goal
            ...recommendedSkills.map((_, i) => ({
                source: currentSkills.length + 2 + i,
                target: nodes.length - 1,
                value: 1
            }))
        ];
        
        const sankeyData = {
            type: "sankey",
            orientation: "h",
            node: {
                pad: 15,
                thickness: 20,
                line: { color: "black", width: 0.5 },
                label: nodes.map(n => n.name)
            },
            link: {
                source: links.map(l => l.source),
                target: links.map(l => l.target),
                value: links.map(l => l.value)
            }
        };
        
        const layout = {
            title: "Skill Acquisition Path",
            font: { size: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 25, l: 25, r: 25, b: 25 }
        };
        
        Plotly.newPlot(chartElement, [sankeyData], layout);
    } else {
        // Fallback if Plotly isn't available
        chartElement.innerHTML = '<p>Chart visualization library not available.</p>';
    }
}

function initializeSkillNetwork() {
    const networkElement = document.getElementById('skill-network-graph');
    if (!networkElement) return;
    
    // Fetch skill graph data
    fetch('/api/skill-graph')
        .then(response => response.json())
        .then(data => {
            if (window.Plotly) {
                // Create a network visualization
                createNetworkGraph(networkElement, data);
            } else {
                networkElement.innerHTML = '<p>Network visualization library not available.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching skill graph:', error);
            networkElement.innerHTML = '<p>Error loading skill network data.</p>';
        });
}

function createNetworkGraph(element, data) {
    // Simple network visualization with Plotly
    if (!data.nodes || !data.edges || data.nodes.length === 0) {
        element.innerHTML = '<p>No skill relationship data available.</p>';
        return;
    }
    
    // Create a simple force-directed layout
    const nodes = data.nodes;
    const edges = data.edges;
    
    // Create nodes trace
    const nodeTrace = {
        x: nodes.map((_, i) => Math.cos(2 * Math.PI * i / nodes.length)),
        y: nodes.map((_, i) => Math.sin(2 * Math.PI * i / nodes.length)),
        mode: 'markers+text',
        marker: {
            size: nodes.map(n => n.size || 15),
            color: '#0055ff'
        },
        text: nodes.map(n => n.label || n.id),
        hoverinfo: 'text',
        textposition: 'top center'
    };
    
    // Create edge traces
    const edgeTraces = [];
    edges.forEach(edge => {
        const sourceIdx = nodes.findIndex(n => n.id === edge.source);
        const targetIdx = nodes.findIndex(n => n.id === edge.target);
        
        if (sourceIdx >= 0 && targetIdx >= 0) {
            const x0 = Math.cos(2 * Math.PI * sourceIdx / nodes.length);
            const y0 = Math.sin(2 * Math.PI * sourceIdx / nodes.length);
            const x1 = Math.cos(2 * Math.PI * targetIdx / nodes.length);
            const y1 = Math.sin(2 * Math.PI * targetIdx / nodes.length);
            
            edgeTraces.push({
                x: [x0, x1, null],
                y: [y0, y1, null],
                mode: 'lines',
                line: {
                    width: edge.weight * 5,
                    color: 'rgba(150, 150, 150, 0.6)'
                },
                hoverinfo: 'none'
            });
        }
    });
    
    const layout = {
        title: 'Skill Relationship Network',
        showlegend: false,
        hovermode: 'closest',
        margin: { t: 50, r: 20, l: 20, b: 20 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot(element, [...edgeTraces, nodeTrace], layout);
}

// Add to app/static/js/main.js
function handleEmptyChart(chartElement, message) {
    Plotly.newPlot(chartElement, [{
        x: [],
        y: [],
        type: 'bar'
    }], {
        annotations: [{
            text: message || "No data available",
            xref: "paper",
            yref: "paper",
            showarrow: false,
            font: {
                size: 16
            },
            x: 0.5,
            y: 0.5
        }],
        xaxis: {showgrid: false, zeroline: false, showticklabels: false},
        yaxis: {showgrid: false, zeroline: false, showticklabels: false},
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    });
}

// Update event handlers to use this function
document.addEventListener('DOMContentLoaded', function() {
    // Example for rising skills chart
    fetch('/api/rising-skills')
        .then(response => response.json())
        .then(data => {
            const chartElement = document.getElementById('rising-skills-chart');
            if (!chartElement) return;
            
            if (!data.skills || data.skills.length === 0) {
                handleEmptyChart(chartElement, "No rising skills data available");
                return;
            }
            
            Plotly.newPlot(chartElement, [{
                x: data.skills,
                y: data.values,
                type: 'bar',
                marker: {
                    color: '#0055ff'
                }
            }], {
                margin: { t: 10, r: 10, b: 50, l: 50 },
                yaxis: {
                    title: 'Growth Rate',
                    gridcolor: 'rgba(255,255,255,0.1)'
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {
                    color: 'rgba(255,255,255,0.8)'
                }
            });
        })
        .catch(error => {
            console.error('Error fetching rising skills:', error);
            const chartElement = document.getElementById('rising-skills-chart');
            if (chartElement) {
                handleEmptyChart(chartElement, "Error loading data");
            }
        });
});

// Add to app/static/js/main.js
function initializeNetworkGraph() {
    const networkElement = document.getElementById('skill-network-graph');
    if (!networkElement) return;
    
    console.log("Initializing network graph");
    
    // Create a very simple default graph if no data
    const defaultNodes = [
        {id: "Node 1", x: -0.5, y: 0.5},
        {id: "Node 2", x: 0.5, y: 0.5},
        {id: "Node 3", x: 0, y: -0.5}
    ];
    
    const defaultEdges = [
        {source: 0, target: 1},
        {source: 1, target: 2},
        {source: 2, target: 0}
    ];
    
    // Use either real data or default visualization
    try {
        const trace = {
            x: defaultNodes.map(node => node.x),
            y: defaultNodes.map(node => node.y),
            mode: 'markers+text',
            marker: {
                size: 20,
                color: '#0055ff'
            },
            text: defaultNodes.map(node => node.id),
            textposition: 'top center',
            hoverinfo: 'text'
        };
        
        const layout = {
            title: 'Skill Network',
            showlegend: false,
            margin: {t: 50, r: 20, l: 20, b: 20},
            xaxis: {showgrid: false, zeroline: false, showticklabels: false},
            yaxis: {showgrid: false, zeroline: false, showticklabels: false},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };
        
        Plotly.newPlot(networkElement, [trace], layout);
    } catch (error) {
        console.error('Error creating network graph:', error);
        networkElement.innerHTML = '<div class="error-message">Error creating network visualization</div>';
    }
}

// Call this function from your page load handler
document.addEventListener('DOMContentLoaded', function() {
    // For the relationships page
    if (document.querySelector('.relationships-section')) {
        initializeNetworkGraph();
    }
});

// Add to app/static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle Generate Recommendations button
    const generateBtn = document.getElementById('generate-recommendations-button');
    if (generateBtn) {
        generateBtn.addEventListener('click', function() {
            console.log("Generate button clicked");
            
            // Get current skills
            const currentSkills = ["Python", "SQL", "Machine Learning", "Excel"];
            
            // Get target job
            const targetJob = document.getElementById('target-job-dropdown')?.value || "Data Analyst";
            
            // Show result - this is a fallback when no API is available
            const recommendationsContainer = document.getElementById('recommendations-container');
            if (recommendationsContainer) {
                recommendationsContainer.innerHTML = `
                    <div class="recommendation-item">
                        <h4>1. Data Visualization</h4>
                        <p>Trend: Growing</p>
                        <p>Resources: Online courses, Books, Practice projects</p>
                    </div>
                    <div class="recommendation-item">
                        <h4>2. Statistics</h4>
                        <p>Trend: Stable</p>
                        <p>Resources: Online courses, Books, Practice projects</p>
                    </div>
                `;
            }
            
            // Create a simple chart for the path
            const chartElement = document.getElementById('skill-path-chart');
            if (chartElement && window.Plotly) {
                // Create a simple chart
                handleEmptyChart(chartElement, "Career path visualization will appear here");
            }
        });
    }
});
// Add this to the end of main.js
document.addEventListener('DOMContentLoaded', function() {
    // Handle Generate Recommendations button click
    const generateBtn = document.getElementById('generate-recommendations-button');
    if (generateBtn) {
        generateBtn.addEventListener('click', function() {
            // Get selected skills and target job
            const skillsElement = document.querySelector('.current-skills-list');
            const skills = skillsElement ? Array.from(skillsElement.querySelectorAll('li')).map(item => item.textContent) : [];
            
            const targetJob = document.getElementById('target-job-dropdown')?.value;
            const timeHorizon = document.getElementById('time-horizon-dropdown')?.value;
            
            if (!targetJob) {
                alert('Please select a target job role');
                return;
            }
            
            // Show loading state
            const recommendationsContainer = document.getElementById('recommendations-container');
            if (recommendationsContainer) {
                recommendationsContainer.innerHTML = '<div class="loading-spinner"></div>';
            }
            
            const pathChart = document.getElementById('skill-path-chart');
            if (pathChart) {
                pathChart.innerHTML = '<div class="loading-spinner"></div>';
            }
            
            // Fetch recommendations (replace with actual API endpoint)
            fetch(`/api/recommendations?target=${targetJob}&time=${timeHorizon}${skills.map(s => `&skill=${s}`).join('')}`)
                .then(response => response.json())
                .then(data => {
                    // Update recommendations
                    updateRecommendations(data);
                    
                    // Update path chart
                    updateSkillPathChart(data);
                })
                .catch(error => {
                    console.error('Error fetching recommendations:', error);
                    if (recommendationsContainer) {
                        recommendationsContainer.innerHTML = '<p>Error loading recommendations. Please try again.</p>';
                    }
                });
        });
    }
    
    // Initialize skill network graph
    initializeSkillNetwork();
});

function updateRecommendations(data) {
    const container = document.getElementById('recommendations-container');
    if (!container) return;
    
    // If no data or no recommendations
    if (!data || !data.recommendations || data.recommendations.length === 0) {
        container.innerHTML = '<p>No specific recommendations found. You may already have the required skills for this role.</p>';
        return;
    }
    
    // Create recommendations HTML
    let html = '';
    data.recommendations.forEach((rec, index) => {
        html += `
            <div class="recommendation-item">
                <h4>${index + 1}. ${rec.skill}</h4>
                <p>Trend: ${rec.trend}</p>
                <p>Resources: ${rec.resources}</p>
            </div>
        `;
    });
    
    container.innerHTML = html;
}

function updateSkillPathChart(data) {
    const chartElement = document.getElementById('skill-path-chart');
    if (!chartElement) return;
    
    // Create a simple visualization if Plotly is available
    if (window.Plotly) {
        const currentSkills = data.currentSkills || [];
        const recommendedSkills = data.recommendations ? data.recommendations.map(r => r.skill) : [];
        
        // Create Sankey data
        const nodes = [
            { name: "Current Skills" },
            ...currentSkills.map(s => ({ name: s })),
            { name: "Target Skills" },
            ...recommendedSkills.map(s => ({ name: s })),
            { name: data.targetJob || "Career Goal" }
        ];
        
        const links = [
            // Link from "Current Skills" to each current skill
            ...currentSkills.map((_, i) => ({
                source: 0,
                target: i + 1,
                value: 1
            })),
            // Link from "Target Skills" to each recommended skill
            ...recommendedSkills.map((_, i) => ({
                source: currentSkills.length + 1,
                target: currentSkills.length + 2 + i,
                value: 1
            })),
            // Link from each recommended skill to the career goal
            ...recommendedSkills.map((_, i) => ({
                source: currentSkills.length + 2 + i,
                target: nodes.length - 1,
                value: 1
            }))
        ];
        
        const sankeyData = {
            type: "sankey",
            orientation: "h",
            node: {
                pad: 15,
                thickness: 20,
                line: { color: "black", width: 0.5 },
                label: nodes.map(n => n.name)
            },
            link: {
                source: links.map(l => l.source),
                target: links.map(l => l.target),
                value: links.map(l => l.value)
            }
        };
        
        const layout = {
            title: "Skill Acquisition Path",
            font: { size: 10 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 25, l: 25, r: 25, b: 25 }
        };
        
        Plotly.newPlot(chartElement, [sankeyData], layout);
    } else {
        // Fallback if Plotly isn't available
        chartElement.innerHTML = '<p>Chart visualization library not available.</p>';
    }
}

function initializeSkillNetwork() {
    const networkElement = document.getElementById('skill-network-graph');
    if (!networkElement) return;
    
    // Fetch skill graph data
    fetch('/api/skill-graph')
        .then(response => response.json())
        .then(data => {
            if (window.Plotly) {
                // Create a network visualization
                createNetworkGraph(networkElement, data);
            } else {
                networkElement.innerHTML = '<p>Network visualization library not available.</p>';
            }
        })
        .catch(error => {
            console.error('Error fetching skill graph:', error);
            networkElement.innerHTML = '<p>Error loading skill network data.</p>';
        });
}

function createNetworkGraph(element, data) {
    // Simple network visualization with Plotly
    if (!data.nodes || !data.edges || data.nodes.length === 0) {
        element.innerHTML = '<p>No skill relationship data available.</p>';
        return;
    }
    
    // Create a simple force-directed layout
    const nodes = data.nodes;
    const edges = data.edges;
    
    // Create nodes trace
    const nodeTrace = {
        x: nodes.map((_, i) => Math.cos(2 * Math.PI * i / nodes.length)),
        y: nodes.map((_, i) => Math.sin(2 * Math.PI * i / nodes.length)),
        mode: 'markers+text',
        marker: {
            size: nodes.map(n => n.size || 15),
            color: '#0055ff'
        },
        text: nodes.map(n => n.label || n.id),
        hoverinfo: 'text',
        textposition: 'top center'
    };
    
    // Create edge traces
    const edgeTraces = [];
    edges.forEach(edge => {
        const sourceIdx = nodes.findIndex(n => n.id === edge.source);
        const targetIdx = nodes.findIndex(n => n.id === edge.target);
        
        if (sourceIdx >= 0 && targetIdx >= 0) {
            const x0 = Math.cos(2 * Math.PI * sourceIdx / nodes.length);
            const y0 = Math.sin(2 * Math.PI * sourceIdx / nodes.length);
            const x1 = Math.cos(2 * Math.PI * targetIdx / nodes.length);
            const y1 = Math.sin(2 * Math.PI * targetIdx / nodes.length);
            
            edgeTraces.push({
                x: [x0, x1, null],
                y: [y0, y1, null],
                mode: 'lines',
                line: {
                    width: edge.weight * 5,
                    color: 'rgba(150, 150, 150, 0.6)'
                },
                hoverinfo: 'none'
            });
        }
    });
    
    const layout = {
        title: 'Skill Relationship Network',
        showlegend: false,
        hovermode: 'closest',
        margin: { t: 50, r: 20, l: 20, b: 20 },
        xaxis: { showgrid: false, zeroline: false, showticklabels: false },
        yaxis: { showgrid: false, zeroline: false, showticklabels: false },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)'
    };
    
    Plotly.newPlot(element, [...edgeTraces, nodeTrace], layout);
}