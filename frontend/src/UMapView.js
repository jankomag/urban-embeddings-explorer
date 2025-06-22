import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

const UMapView = ({ locations, selectedLocations, onLocationSelect }) => {
  const svgRef = useRef(null);
  const [umapData, setUmapData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

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
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Fetch UMAP data when component mounts
  useEffect(() => {
    if (locations.length > 0) {
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

  const fetchUmapData = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/umap`);
      if (!response.ok) {
        throw new Error('Failed to fetch UMAP data');
      }
      const data = await response.json();
      setUmapData(data);
    } catch (err) {
      console.error('Error fetching UMAP data:', err);
      setError('Failed to load UMAP visualization. Computing embeddings...');
      
      // Fallback: compute UMAP on frontend (simplified version)
      await computeSimpleUmap();
    } finally {
      setLoading(false);
    }
  };

  const computeSimpleUmap = async () => {
    try {
      // This is a simplified 2D projection for demo purposes
      // In production, you'd want proper UMAP computation on the backend
      const data = locations.map((location, index) => {
        // Create a simple 2D projection based on geographic coordinates
        // with some random jitter to simulate embedding space
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
          latitude: location.latitude
        };
      });

      setUmapData({ umap_points: data });
    } catch (err) {
      setError('Failed to compute UMAP visualization');
    }
  };

  // Create visualization
  useEffect(() => {
    if (!umapData || !svgRef.current) return;

    createVisualization();
  }, [umapData, dimensions, selectedLocations]);

  const createVisualization = () => {
    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const { width, height } = dimensions;
    const margin = { top: 20, right: 20, bottom: 60, left: 60 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    // Create scales
    const xExtent = d3.extent(umapData.umap_points, d => d.x);
    const yExtent = d3.extent(umapData.umap_points, d => d.y);
    
    const xScale = d3.scaleLinear()
      .domain(xExtent)
      .range([0, plotWidth])
      .nice();

    const yScale = d3.scaleLinear()
      .domain(yExtent)
      .range([plotHeight, 0])
      .nice();

    // Create container group
    const container = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add background
    container.append('rect')
      .attr('width', plotWidth)
      .attr('height', plotHeight)
      .attr('fill', '#f8f9fa')
      .attr('stroke', '#dee2e6')
      .attr('rx', 4);

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
      .text('UMAP Dimension 1');

    container.append('text')
      .attr('class', 'axis-label')
      .attr('text-anchor', 'middle')
      .attr('transform', 'rotate(-90)')
      .attr('x', -plotHeight / 2)
      .attr('y', -40)
      .text('UMAP Dimension 2');

    // Create tooltip
    const tooltip = d3.select('body').append('div')
      .attr('class', 'umap-tooltip')
      .style('opacity', 0)
      .style('position', 'absolute')
      .style('background', 'rgba(0, 0, 0, 0.8)')
      .style('color', 'white')
      .style('padding', '8px')
      .style('border-radius', '4px')
      .style('pointer-events', 'none')
      .style('font-size', '12px')
      .style('z-index', '1000');

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
      .on('mouseover', function(event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('r', 6)
          .attr('opacity', 1);

        tooltip.transition()
          .duration(200)
          .style('opacity', .9);
        
        tooltip.html(`
          <strong>${d.city}</strong><br/>
          ${d.country}<br/>
          <em>${d.continent}</em>
        `)
          .style('left', (event.pageX + 10) + 'px')
          .style('top', (event.pageY - 28) + 'px');
      })
      .on('mouseout', function(event, d) {
        d3.select(this)
          .transition()
          .duration(150)
          .attr('r', selectedLocations.has(d.location_id) ? 6 : 4)
          .attr('opacity', 0.8);

        tooltip.transition()
          .duration(500)
          .style('opacity', 0);
      })
      .on('click', function(event, d) {
        onLocationSelect(d.location_id);
      });

    // Add legend
    const legend = container.append('g')
      .attr('class', 'legend')
      .attr('transform', `translate(${plotWidth - 120}, 20)`);

    const continents = Object.keys(continentColors);
    const legendItems = legend.selectAll('.legend-item')
      .data(continents)
      .enter()
      .append('g')
      .attr('class', 'legend-item')
      .attr('transform', (d, i) => `translate(0, ${i * 20})`);

    legendItems.append('circle')
      .attr('cx', 8)
      .attr('cy', 8)
      .attr('r', 5)
      .attr('fill', d => continentColors[d]);

    legendItems.append('text')
      .attr('x', 18)
      .attr('y', 8)
      .attr('dy', '0.35em')
      .style('font-size', '11px')
      .style('fill', '#333')
      .text(d => d);

    // Clean up tooltip on component unmount
    return () => {
      tooltip.remove();
    };
  };

  const highlightPoint = (locationId) => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    
    // Reset all points
    svg.selectAll('.umap-point')
      .transition()
      .duration(300)
      .attr('r', d => selectedLocations.has(d.location_id) ? 6 : 4)
      .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1)
      .attr('opacity', 0.8);

    // Highlight specific point
    svg.selectAll('.umap-point')
      .filter(d => d.location_id === locationId)
      .transition()
      .duration(300)
      .attr('r', 8)
      .attr('stroke', '#ffd700')
      .attr('stroke-width', 4)
      .attr('opacity', 1);

    // Reset highlight after delay
    setTimeout(() => {
      svg.selectAll('.umap-point')
        .filter(d => d.location_id === locationId)
        .transition()
        .duration(500)
        .attr('r', d => selectedLocations.has(d.location_id) ? 6 : 4)
        .attr('stroke', d => selectedLocations.has(d.location_id) ? '#ff6b6b' : '#fff')
        .attr('stroke-width', d => selectedLocations.has(d.location_id) ? 3 : 1)
        .attr('opacity', 0.8);
    }, 2000);
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
        <p>Points colored by continent • Click to select • Hover for details</p>
      </div>
      <div className="umap-plot">
        <svg ref={svgRef}></svg>
      </div>
    </div>
  );
};

export default UMapView;