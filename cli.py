import asyncio
import json
import click
import aiohttp
from datetime import datetime, timedelta
from typing import Optional

from .models.log_entry import LogEntry, LogLevel
from .models.search import SearchQuery


@click.group()
def cli():
    pass


@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def health(host: str, port: int):
    async def check_health():
        url = f"http://{host}:{port}/health/detailed"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo("Ghost tracer is healthy!")
                        click.echo(json.dumps(data, indent=2))
                    else:
                        click.echo(f"Health check failed: {response.status}")
        except Exception as e:
            click.echo(f"Connection failed: {e}")
    
    asyncio.run(check_health())


@cli.command()
@click.option('--message', required=True, help='Log message')
@click.option('--service', required=True, help='Service name')
@click.option('--level', default='INFO', help='Log level')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def send_log(message: str, service: str, level: str, host: str, port: int):
    async def send():
        log_entry = LogEntry(
            timestamp=datetime.utcnow(),
            level=LogLevel(level.upper()),
            message=message,
            service_name=service
        )
        
        url = f"http://{host}:{port}/api/v1/logs/ingest"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, 
                    json=log_entry.dict(),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo(f"Log sent successfully: {data['log_id']}")
                    else:
                        click.echo(f"Failed to send log: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(send())


@cli.command()
@click.option('--query', required=True, help='Search query')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--limit', default=10, help='Number of results')
def search(query: str, host: str, port: int, limit: int):
    async def search_logs():
        search_query = SearchQuery(
            query_text=query,
            limit=limit
        )
        
        url = f"http://{host}:{port}/api/v1/search"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=search_query.dict(),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo(f"Found {data['total_count']} results:")
                        
                        for item in data['items']:
                            click.echo(f"  🕐 {item['timestamp']}")
                            click.echo(f"  📝 {item['service_name']}: {item['message']}")
                            click.echo(f"  🎯 Relevance: {item['relevance_score']:.2f}")
                            click.echo("  ---")
                        
                        if data.get('ai_summary'):
                            click.echo(f"\nAI Summary:\n{data['ai_summary']}")
                    else:
                        click.echo(f"Search failed: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(search_logs())


@cli.command()
@click.option('--query', required=True, help='Analysis query')
@click.option('--hours', default=1, help='Hours to analyze (from now)')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def analyze(query: str, hours: int, host: str, port: int):
    async def analyze_logs():
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        url = f"http://{host}:{port}/api/v1/analysis/root-cause"
        params = {
            'query': query,
            'time_start': start_time.isoformat(),
            'time_end': end_time.isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo(f"Root cause analysis for: {query}")
                        click.echo(f"Time range: {start_time} to {end_time}")
                        click.echo(f"Analyzed {data['analyzed_logs_count']} logs")
                        click.echo(f"\nSummary:\n{data['summary']}")
                    else:
                        click.echo(f"Analysis failed: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(analyze_logs())


@cli.command()
@click.option('--hours', default=1, help='Hours to summarize (from now)')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def summarize(hours: int, host: str, port: int):
    async def summarize_logs():
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        url = f"http://{host}:{port}/api/v1/analysis/summarize"
        params = {
            'time_start': start_time.isoformat(),
            'time_end': end_time.isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo(f"Log summary for last {hours} hour(s)")
                        click.echo(f"Time range: {start_time} to {end_time}")
                        click.echo(f"Processed {data['total_logs_processed']} logs")
                        click.echo(f"\nSummary:\n{data['summary']}")
                    else:
                        click.echo(f"Summarization failed: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(summarize_logs())


@cli.command()
@click.option('--query', required=True, help='Chat query')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def chat(query: str, host: str, port: int):
    async def chat_with_logs():
        url = f"http://{host}:{port}/api/v1/analysis/chat"
        params = {'query': query}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        click.echo(f"💬 You: {query}")
                        click.echo(f"🤖 ghost tracer: {data['response']}")
                    else:
                        click.echo(f"Chat failed: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(chat_with_logs())


@cli.command()
@click.option('--window-minutes', default=30, help='Time window to analyze in minutes')
@click.option('--services', help='Comma-separated list of services to analyze')
@click.option('--sensitivity', default=0.7, help='Anomaly sensitivity (0.1-1.0)')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
def detect_anomalies(window_minutes: int, services: str, sensitivity: float, host: str, port: int):
    """Detect anomalies in system logs using AI and statistical analysis"""
    async def run_anomaly_detection():
        url = f"http://{host}:{port}/api/v1/analysis/anomaly-detection"
        params = {
            'time_window_minutes': window_minutes,
            'sensitivity': sensitivity
        }
        
        if services:
            service_list = [s.strip() for s in services.split(',')]
            params['services'] = service_list
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        click.echo(f"Anomaly Detection Results")
                        click.echo(f"Time Window: {data['time_window']['duration_minutes']} minutes")
                        click.echo(f"System Health Score: {data['system_health_score']:.2f}/1.0")
                        click.echo(f"Anomalies Detected: {data['anomalies_detected']}")
                        click.echo(f"High Severity: {data['high_severity_count']}")
                        click.echo()
                        
                        if data['anomalies']:
                            click.echo("🚨 Detected Anomalies:")
                            for i, anomaly in enumerate(data['anomalies'], 1):
                                severity_emoji = "🔴" if anomaly.get('severity_score', 0) > 0.7 else "🟡" if anomaly.get('severity_score', 0) > 0.4 else "🟢"
                                click.echo(f"{severity_emoji} {i}. {anomaly['title']}")
                                click.echo(f"   {anomaly['description']}")
                                click.echo(f"   Severity: {anomaly.get('severity_score', 0):.2f}")
                                if 'change_percentage' in anomaly:
                                    click.echo(f"   Change: {anomaly['change_percentage']:+.1f}%")
                                click.echo()
                        else:
                            click.echo("No anomalies detected - system appears healthy")
                        
                        if data['recommendations']:
                            click.echo("Recommendations:")
                            for rec in data['recommendations']:
                                click.echo(f"   {rec}")
                        
                    else:
                        click.echo(f"Anomaly detection failed: {response.status}")
        except Exception as e:
            click.echo(f"Error: {e}")
    
    asyncio.run(run_anomaly_detection())


if __name__ == '__main__':
    cli() 