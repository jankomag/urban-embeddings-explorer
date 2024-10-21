import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import {
  Chart as ChartJS,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController
} from 'chart.js';
import zoomPlugin from 'chartjs-plugin-zoom';

ChartJS.register(
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  ScatterController,
  zoomPlugin
);

const ScatterPlot = React.memo(({ onPointSelect, selectedPoint, selectedCountry, selectedCity }) => {
  const [data, setData] = useState([]);
  const [isFlashing, setIsFlashing] = useState(false);
  const canvasRef = useRef(null);
  const chartRef = useRef(null);
  const countryColors = useRef({});
  const flashIntervalRef = useRef(null);
  const [dataRange, setDataRange] = useState({ xMin: 0, xMax: 0, yMin: 0, yMax: 0 });

  const formatCountryName = (name) => {
    return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()).join(' ');
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:8000/tsne_data');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        console.log(`Received ${result.data.length} data points`);
        setData(result.data);

        // Calculate data range
        const xValues = result.data.map(d => d.x);
        const yValues = result.data.map(d => d.y);
        setDataRange({
          xMin: Math.min(...xValues),
          xMax: Math.max(...xValues),
          yMin: Math.min(...yValues),
          yMax: Math.max(...yValues)
        });

        result.data.forEach(d => {
          if (!countryColors.current[d.country]) {
            countryColors.current[d.country] = `hsla(${Math.random() * 360}, 70%, 50%, 0.5)`;            
          }
        });
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const chartData = useMemo(() => {
    console.log("Selected point in ScatterPlot:", selectedPoint);
    const baseDataset = {
      label: 'Base Points',
      data: data.filter(d => {
        if (selectedPoint) return d.x !== selectedPoint.x || d.y !== selectedPoint.y;
        if (selectedCity) return d.city !== selectedCity;
        return true;
      }).map(d => ({
        ...d,
        country: formatCountryName(d.country)
      })),
      backgroundColor: data.map(d => countryColors.current[d.country] || 'rgba(128, 128, 128, 0.5)'),
      borderColor: 'transparent',
      borderWidth: 0,
      pointRadius: 2,
      pointHoverRadius: 4,
    };

    const highlightedDataset = {
      label: 'Highlighted Points',
      data: data.filter(d => {
        if (selectedPoint) return d.x === selectedPoint.x && d.y === selectedPoint.y;
        if (selectedCity) return d.city === selectedCity;
        return false;
      }).map(d => ({
        ...d,
        country: formatCountryName(d.country)
      })),
      backgroundColor: isFlashing ? 'rgba(255, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.6)',
      borderColor: 'black',
      borderWidth: 1,
      pointRadius: 6,
      pointHoverRadius: 6,
    };

    return {
      datasets: [baseDataset, highlightedDataset].reverse(),
    };
  }, [data, selectedPoint, selectedCity, isFlashing]);

  const chartOptions = useMemo(() => {
    const padding = 0.1; // 10% padding
    const xRange = dataRange.xMax - dataRange.xMin;
    const yRange = dataRange.yMax - dataRange.yMin;
    const xPadding = xRange * padding;
    const yPadding = yRange * padding;

    return {
      scales: {
        x: { 
          type: 'linear', 
          position: 'bottom',
          min: dataRange.xMin - xPadding,
          max: dataRange.xMax + xPadding,
        },
        y: { 
          type: 'linear', 
          position: 'left',
          min: dataRange.yMin - yPadding,
          max: dataRange.yMax + yPadding,
        }
      },
      plugins: {
        tooltip: {
          enabled: true,
          mode: 'nearest',
          intersect: false,
          callbacks: {
            label: (context) => {
              const point = context.raw;
              return `Country: ${point.country}, City: ${point.city || 'N/A'}`;
            },
            title: () => null
          }
        },
        legend: {
          display: false
        },
        zoom: {
          pan: {
            enabled: true,
            mode: 'xy',
          },
          zoom: {
            wheel: {
              enabled: true,
            },
            pinch: {
              enabled: true
            },
            mode: 'xy',
          },
          limits: {
            x: {min: dataRange.xMin - xPadding, max: dataRange.xMax + xPadding},
            y: {min: dataRange.yMin - yPadding, max: dataRange.yMax + yPadding}
          }
        }
      },
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      onClick: (event, elements) => {
        if (elements.length > 0) {
          const index = elements[0].index;
          const clickedPoint = data[index];
          console.log("Scatter plot clicked, point:", clickedPoint);
          onPointSelect(clickedPoint);
        }
      },
    };
  }, [data, onPointSelect, dataRange]);

  const centerOnPoint = useCallback((point) => {
    if (chartRef.current) {
      const chart = chartRef.current;
      const xScale = chart.scales.x;
      const yScale = chart.scales.y;
      
      const xCenter = (xScale.max + xScale.min) / 2;
      const yCenter = (yScale.max + yScale.min) / 2;
      
      const xOffset = point.x - xCenter;
      const yOffset = point.y - yCenter;
      
      chart.pan({x: -xOffset * xScale.width / (xScale.max - xScale.min), y: -yOffset * yScale.height / (yScale.max - yScale.min)}, undefined, 'default');
      chart.update('none');
    }
  }, []);

  useEffect(() => {
    if (data.length > 0 && canvasRef.current) {
      if (chartRef.current) {
        chartRef.current.destroy();
      }

      const ctx = canvasRef.current.getContext('2d');
      chartRef.current = new ChartJS(ctx, {
        type: 'scatter',
        data: chartData,
        options: chartOptions
      });
    }

    return () => {
      if (chartRef.current) {
        chartRef.current.destroy();
      }
    };
  }, [data, chartData, chartOptions]);

  useEffect(() => {
    if (chartRef.current && selectedPoint) {
      centerOnPoint(selectedPoint);
    }
  }, [selectedPoint, centerOnPoint]);

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.resetZoom();
      chartRef.current.update();
      console.log("Zoom reset");
    }
  }, []);

  const handleFlashPoints = useCallback(() => {
    if (flashIntervalRef.current) {
      clearInterval(flashIntervalRef.current);
    }

    let flashCount = 0;
    flashIntervalRef.current = setInterval(() => {
      setIsFlashing(prev => !prev);
      flashCount++;
      if (flashCount >= 4) { // Flash 5 times (10 state changes)
        clearInterval(flashIntervalRef.current);
        setIsFlashing(false);
      }
    }, 50); // Change every 50ms for a faster flash
  }, []);

  useEffect(() => {
    return () => {
      if (flashIntervalRef.current) {
        clearInterval(flashIntervalRef.current);
      }
    };
  }, []);

  return (
    <div style={{ height: '100%', width: '100%', position: 'relative' }}>
      <canvas ref={canvasRef} style={{ height: '100%', width: '100%' }} />
      <div style={{ position: 'absolute', top: '10px', right: '10px', zIndex: 10 }}>
        <button 
          onClick={handleResetZoom}
          style={{
            padding: '5px 10px',
            backgroundColor: '#4CAF50',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '12px',
            marginRight: '10px'
          }}
        >
          Reset Zoom
        </button>
        {(selectedPoint || selectedCity) && (
          <button 
            onClick={handleFlashPoints}
            style={{
              padding: '5px 10px',
              backgroundColor: '#4CAF50',
              color: 'white',
              border: 'none',
              borderRadius: '5px',
              cursor: 'pointer',
              fontSize: '12px'
            }}
          >
            Flash Selected Point(s)
          </button>
        )}
      </div>
    </div>
  );
});

export default ScatterPlot;