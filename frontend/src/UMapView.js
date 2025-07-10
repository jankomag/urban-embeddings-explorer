import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

const UMapView = ({ locations, selectedLocations, onLocationSelect }) => {
  const svgRef = useRef(null);
  const [umapData, setUmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });
  const [activeTooltip, setActiveTooltip] = useState(null);
  
  // Refs for D3 elements
  const zoomRef = useRef(null);
  const tooltipRef = useRef(null);
  const containerRef = useRef(null);

  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://your-domain.com'
    : 'http://localhost:8000';

  // Continent color mapping
  const continentColors = {
    'Africa': '#e74c3c',
    'Asia': '#3498db',
    'Europe': '#2ecc71',
    'North America': '#f39c12',
    'South America': '#9b59b6',
    'Oceania': '#1abc9c',
    'Antarctica': '#95a5a6'
  };

  // Update dimensions on window resize
  useEffect(() => {
    const updateDimensions = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        const rect = container.getBoundingClientRect();
        setDimensions({
          width: Math.max(400, rect.width - 40),
          height: Math.max(300, rect.height - 40)
        });
      }
    };

    updateDimensions();
    const resizeHandler = () => {
      updateDimensions();
    };
    
    window.addEventListener('resize', resizeHandler);
    return () => window.removeEventListener('resize', resizeHandler);
  }, []);

  // Fetch UMAP data when component mounts
  useEffect(() => {
    if (locations.length > 0 && !umapData) {
      fetchUmapData();
    }
  }, [locations]);

  // Listen for highlight events from similarity panel
  useEffect(() => {
    const handleHighlightPoint = (event) => {
      const { locationId } = event.detail;
      highlightPoint(locationId);
    };

    window.addEventListener('highlightUmapPoint', handleHighlightPoint);
    return () => window.removeEventListener('highlightUmapPoint', handleHighlightPoint);
  }, [umapData]);

  // Create tooltip on mount
  useEffect(() => {
    if (!tooltipRef.current) {
      tooltipRef.current = d3.select('body').append('div')
        .attr('class', 'umap-tooltip')
        .style('opacity', 0)
        .style('position', 'absolute')
        .style('background', 'rgba(0, 0, 0, 0.9)')
        .style('color', 'white')
        .style('padding', '12px')
        .style('border-radius', '8px')
        .style('pointer-events', 'none')
        .style('font-size', '13px')
        .style('z-index', '1000')
        .style('box-shadow', '0 4px 12px rgba(0,0,0,0.3)')
        .style('border', '1px solid rgba(255,255,255,0.2)')
        .style('max-width', '200px');
    }

    return () => {
      if (tooltipRef.current) {
        tooltipRef.current.remove();
        tooltipRef.current = null;
      }
    };
  }, []);

  const fetchUmapData = async () => {
    setLoading(true);
    setError(null);

    try {
      console.log('Fetching UMAP data...');
      
      const response = await fetch(`${API_BASE_URL}/api/umap`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch UMAP data');
      }
      const data = await response.json();
      setUmapData(data);
      console.log(`UMAP data loaded: ${data.total_points} points`);
    } catch (err) {
      console.error('Error fetching UMAP data:', err);
      setError(`Failed to load UMAP visualization: ${err.message}`);
      
      // Fallback: compute simple projection on frontend
      await computeSimpleProjection();
    } finally {
      setLoading(false);
    }
  };

  const computeSimpleProjection = async () => {
    try {
      // Create a simple 2D projection based on geographic coordinates
      const data = locations.map((location, index) => {
        const geoProjX = (location.longitude + 180) / 360;
        const geoProjY = (location.latitude + 90) / 180;
        
        // Add some noise to simulate embedding clustering
        const noise = () => (Math.random() - 0.5) * 0.3;
        
        return {
          location_id: location.id,
          x: geoProjX + noise(),
          y: geoProjY + noise(),
          city: location.city,
          country: location.country,
          continent: location.continent,
          longitude: location.longitude,
          latitude: location.latitude,
          date: location.date
        };
      });

      setUmapData({ 
        umap_points: data,
        total_points: data.length
      });
    } catch (err) {
      setError('Failed to compute UMAP visualization');
    }
  };

  // Create visualization
  useEffect(() => {
    if (!umapData || !svgRef.current) return;

    createVisualization();
  }, [umapData, dimensions, selectedLocations]);

  const hideTooltip = () => {
    if (tooltipRef.current) {
      tooltipRef.current.transition()
        .duration(200)
        .style('opacity', 0);
    }
    setActiveTooltip(null);
  };

  const showTooltip = (event, d) => {
    if (!tooltipRef.current) return;

    const tooltip = tooltipRef.current;
    
    tooltip.html(`
      <div style="font-weight: bold; margin-bottom: 6px; color: #4ecdc4;">
        ${d.city}
      </div>
      <div style="margin-bottom: 3px;">
        <span style="color: #ccc;">Country:</span> ${d.country}
      </div>
      <div style="margin-bottom: 3px;">
        <span style="color: #ccc;">Continent:</span> ${d.continent}
      </div>
      ${d.date ? `<div style="margin-bottom: 3px;">
        <span style="color: #ccc;">Date:</span> ${d.date}
      </div>` : ''}
      <div style="margin-top: 6px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.2); font-size: 11px; color: #aaa;">
        Click to select
      </div>
    `);

    // Position tooltip
    const [mouseX, mouseY] = d3.pointer(event, document.body);
    const tooltipNode = tooltip.node();
    const tooltipRect = tooltipNode.getBoundingClientRect();
    
    let left = mouseX + 15;
    let top = mouseY - 10;
    
    // Adjust if tooltip would go off screen
    if (left + tooltipRect.width > window.innerWidth) {
      left = mouseX - tooltipRect.width - 15;
    }
    if (top + tooltipRect.height > window.innerHeight) {
      top = mouseY - tooltipRect.height - 10;
    }
    if (top < 0) {
      top = mouseY + 15;
    }

    tooltip
      .style('left', left + 'px')
      .style('top', top + 'px')
      .transition()
      .duration(200)
      .style('opacity', 0.95);

    setActiveTooltip(d.location_id);
  };

  const createVisualization = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { width, height } = dimensions;
    const margin = { top: 20, right: 160, bottom: 60, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Create scales
    const xExtent = d3.extent(umapData.umap_points, d => d.x);
    const yExtent = d3.extent(umapData.umap_points, d => d.y);
    
    // Add some padding to the domain
    const xPadding = (xExtent[1] - xExtent[0]) * 0.1;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.1;
    
    const xScale = d3.scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([0, plotWidth]);

    const yScale = d3.scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([plotHeight, 0]);

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 10])
      .on('zoom', (event) => {
        const { transform } = event;
        
        // Update scales with zoom transform
        const zoomedXScale = transform.rescaleX(xScale);
        const zoomedYScale = transform.rescaleY(yScale);
        
        // Update axes
        container.select('.x-axis')
          .call(d3.axisBottom(zoomedXScale).ticks(6));
        
        container.select('.y-axis')
          .call(d3.axisLeft(zoomedYScale).ticks(6));
        
        // Update points
        container.selectAll('.umap-point')
          .attr('cx', d => zoomedXScale(d.x))
          .attr('cy', d => zoomedYScale(d.y));
        
        // Hide tooltip during zoom
        hideTooltip();
      });

    zoomRef.current = zoom;

    // Create container group
    const container = svg
      .attr('width', width)
      .attr('height', height)
      .call(zoom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    containerRef.current = container;

    // Add background for zoom area
    container.append('rect')
      .attr('width', plotWidth)
      .attr('height', plotHeight)
      .attr('fill', '#f8f9fa')
      .attr('stroke', '#dee2e6')
      .attr('rx', 4)
      .style('cursor', 'grab')
      .on('click', () => {
        // Hide tooltip when clicking on background
        hideTooltip();
      });

    // Add axes
    const xAxis = d3.axisBottom(xScale).ticks(6);
    const yAxis = d3.axisLeft(yScale).ticks(6);

    container.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${plotHeight})`)
      .call(xAxis);

    container.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);

    // Add axis labels
    container.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('x', plotWidth / 2)
      .attr('y', plotHeight + 40)
      .style('font-size', '12px')
      .style('fill', '#666')
      .text('UMAP Dimension 1');

    container.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -plotHeight / 2)
      .attr('y', -40)
      .style('font-size', '12px')
      .style('fill', '#666')
      .text('UMAP Dimension 2');

    // Add points
    const points = container.selectAll('.umap-point')
      .data(umapData.umap_points)
      .enter()
      .append('circle')
      .attr('class', 'umap-point')
      .attr('cx', d => xScale(d.x))
      .attr('cy', d => yScale(d.y))
      .attr('r', d => selectedLocations.has(d.location_id) ? 6 : 4)
      .attr('fill', d => continentColors[d.continent] || '#666')
      .attr('stroke', d => selectedLocations.has(d.location_id) ? '#ff6b6b' : '#fff')
      .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1)
      .attr('opacity', 0.8)
      .style('cursor', 'pointer')
      .on('mouseenter', function(event, d) {
        if (activeTooltip !== d.location_id) {
          d3.select(this)
            .transition()
            .duration(150)
            .attr('r', selectedLocations.has(d.location_id) ? 8 : 6)
            .attr('opacity', 1)
            .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 4 : 2);
        }
      })
      .on('mouseleave', function(event, d) {
        if (activeTooltip !== d.location_id) {
          d3.select(this)
            .transition()
            .duration(150)
            .attr('r', selectedLocations.has(d.location_id) ? 6 : 4)
            .attr('opacity', 0.8)
            .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1);
        }
      })
      .on('click', function(event, d) {
        event.stopPropagation();
        
        // Hide any existing tooltip
        hideTooltip();
        
        // Show tooltip for clicked point
        setTimeout(() => {
          showTooltip(event, d);
        }, 10);
        
        // Select the location
        onLocationSelect(d.location_id);
        
        // Update visual state immediately
        d3.select(this)
          .transition()
          .duration(200)
          .attr('r', 8)
          .attr('stroke', '#ff6b6b')
          .attr('stroke-width', 4)
          .attr('opacity', 1);
      });

    // Add legend
    const legend = container.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${plotWidth + 20}, 20)`);

    // Legend background
    const continents = Object.keys(continentColors);
    const legendHeight = continents.length * 22 + 20;
    
    legend.append('rect')
      .attr('x', -10)
      .attr('y', -10)
      .attr('width', 130)
      .attr('height', legendHeight)
      .attr('fill', 'rgba(255, 255, 255, 0.95)')
      .attr('stroke', '#dee2e6')
      .attr('rx', 4);

    legend.append('text')
      .attr('x', 0)
      .attr('y', 0)
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .style('fill', '#333')
      .text('Continents');

    const legendItems = legend.selectAll('.legend-item')
      .data(continents)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 22 + 20})`);

    legendItems.append('circle')
      .attr('cx', 8)
      .attr('cy', 8)
      .attr('r', 6)
      .attr('fill', d => continentColors[d])
      .attr('stroke', '#fff')
      .attr('stroke-width', 1);

    legendItems.append('text')
      .attr('x', 20)
      .attr('y', 8)
      .attr('dy', '0.35em')
      .style('font-size', '11px')
      .style('fill', '#333')
      .text(d => d);

    // Add zoom controls
    const zoomControls = container.append('g')
      .attr('class', 'zoom-controls')
      .attr('transform', `translate(${plotWidth - 60}, 20)`);

    // Zoom in button
    const zoomInBtn = zoomControls.append('g')
      .attr('class', 'zoom-btn')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(300).call(
          zoom.scaleBy, 1.5
        );
      });

    zoomInBtn.append('rect')
      .attr('width', 25)
      .attr('height', 25)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#666')
      .attr('rx', 3);

    zoomInBtn.append('text')
      .attr('x', 12.5)
      .attr('y', 12.5)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .style('fill', '#333')
      .text('+');

    // Zoom out button
    const zoomOutBtn = zoomControls.append('g')
      .attr('class', 'zoom-btn')
      .attr('transform', 'translate(0, 30)')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(300).call(
          zoom.scaleBy, 0.67
        );
      });

    zoomOutBtn.append('rect')
      .attr('width', 25)
      .attr('height', 25)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#666')
      .attr('rx', 3);

    zoomOutBtn.append('text')
      .attr('x', 12.5)
      .attr('y', 12.5)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('font-size', '16px')
      .style('font-weight', 'bold')
      .style('fill', '#333')
      .text('−');

    // Reset zoom button
    const resetBtn = zoomControls.append('g')
      .attr('class', 'zoom-btn')
      .attr('transform', 'translate(0, 60)')
      .style('cursor', 'pointer')
      .on('click', () => {
        svg.transition().duration(500).call(
          zoom.transform,
          d3.zoomIdentity
        );
        hideTooltip();
      });

    resetBtn.append('rect')
      .attr('width', 25)
      .attr('height', 25)
      .attr('fill', 'rgba(255, 255, 255, 0.9)')
      .attr('stroke', '#666')
      .attr('rx', 3);

    resetBtn.append('text')
      .attr('x', 12.5)
      .attr('y', 12.5)
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .style('font-size', '10px')
      .style('font-weight', 'bold')
      .style('fill', '#333')
      .text('⌂');
  };

  const highlightPoint = (locationId) => {
    if (!containerRef.current) return;

    // Hide any existing tooltip
    hideTooltip();
    
    // Reset all points
    containerRef.current.selectAll('.umap-point')
      .transition()
      .duration(300)
      .attr('r', d => selectedLocations.has(d.location_id) ? 6 : 4)
      .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1)
      .attr('opacity', 0.8);

    // Highlight specific point
    const targetPoint = containerRef.current.selectAll('.umap-point')
      .filter(d => d.location_id === locationId);

    if (!targetPoint.empty()) {
      targetPoint
        .transition()
        .duration(300)
        .attr('r', 10)
        .attr('stroke', '#ffd700')
        .attr('stroke-width', 4)
        .attr('opacity', 1);

      // Show tooltip for highlighted point
      const pointData = targetPoint.datum();
      if (pointData) {
        // Create a synthetic event for positioning
        const pointNode = targetPoint.node();
        const rect = pointNode.getBoundingClientRect();
        const syntheticEvent = {
          pageX: rect.left + rect.width / 2,
          pageY: rect.top + rect.height / 2
        };
        
        setTimeout(() => {
          showTooltip(syntheticEvent, pointData);
        }, 400);
      }

      // Reset highlight after delay
      setTimeout(() => {
        targetPoint
          .transition()
          .duration(500)
          .attr('r', d => selectedLocations.has(d.location_id) ? 6 : 4)
          .attr('stroke', d => selectedLocations.has(d.location_id) ? '#ff6b6b' : '#fff')
          .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1)
          .attr('opacity', 0.8);
        
        // Keep tooltip if it's the active one
        if (activeTooltip === locationId) {
          hideTooltip();
        }
      }, 2500);
    }
  };

  if (loading) {
    return (
      <div className="umap-loading">
        <div className="spinner"></div>
        <p>Computing UMAP embeddings...</p>
        <small>This may take a moment for large datasets</small>
      </div>
    );
  }

  if (error) {
    return (
      <div className="umap-error">
        <p>{error}</p>
        <button onClick={fetchUmapData} className="retry-btn">
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="umap-container">
      <div className="umap-header">
        <h3>UMAP Embedding Visualization</h3>
        <p>
          TerraMind satellite embeddings projected to 2D space • 
          Points colored by continent • Click to select • Zoom and pan enabled
        </p>
        {umapData && (
          <div className="umap-stats">
            <small>
              {umapData.total_points} points plotted • Scroll to zoom • Drag to pan
            </small>
          </div>
        )}
      </div>
      <div className="umap-plot">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
};

export default UMapView;